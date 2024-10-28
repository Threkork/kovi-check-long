use image::codecs::gif::GifDecoder;
use image::AnimationDecoder;
use image::{imageops::FilterType, GenericImageView};
use image::{DynamicImage, ImageFormat};
use kovi::log::{error, info};
use kovi::{chrono, tokio, AllMsgEvent, Message, RuntimeBot};
use ndarray::{s, Array, Axis};
use ort::GraphOptimizationLevel;
use ort::{inputs, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use std::collections::HashMap;
use std::io::Cursor;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{Config, UserInfo};

#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundingBox {
    pub(crate) x1: f32,
    pub(crate) y1: f32,
    pub(crate) x2: f32,
    pub(crate) y2: f32,
}

#[derive(Clone)]
pub(crate) struct Detector {
    pub(crate) model: Arc<Session>,
    pub(crate) config: Arc<Config>,
    pub(crate) labels: Vec<&'static str>,
    pub(crate) whitelist: Arc<RwLock<HashMap<i64, bool>>>,
    pub(crate) user_info: Arc<Mutex<HashMap<i64, UserInfo>>>,
    pub(crate) data_path: Arc<PathBuf>,
    pub(crate) name: String,
}

impl Detector {
    pub(crate) fn new(
        model_bytes: &[u8],
        config: Config,
        labels: Vec<&'static str>,
        whitelist: Arc<RwLock<HashMap<i64, bool>>>,
        user_info: Arc<Mutex<HashMap<i64, UserInfo>>>,
        data_path: PathBuf,
        name: String,
    ) -> Self {
        let model = Arc::new(
            Session::builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                .with_intra_threads(4)
                .unwrap()
                .commit_from_memory(model_bytes)
                .unwrap(),
        );

        Self {
            model,
            config: Arc::new(config),
            labels,
            whitelist,
            user_info,
            data_path: Arc::new(data_path),
            name,
        }
    }

    pub(crate) fn handle_admin_command(&self, e: Arc<AllMsgEvent>) {
        if e.text.is_none() {
            return;
        }

        let text = e.borrow_text().unwrap();
        if text != self.config.start_cmd && text != self.config.stop_cmd {
            return;
        }

        if !e.is_group() {
            return;
        }

        let mut whitelist = self.whitelist.write().unwrap();
        let group_id = e.group_id.unwrap();

        if text == self.config.start_cmd {
            whitelist.insert(group_id, true);
            e.reply(&self.config.start_msg);
        } else if text == self.config.stop_cmd {
            whitelist.insert(group_id, false);
            e.reply(&self.config.stop_msg);
        }
    }

    pub(crate) fn handle_my_times(&self, e: Arc<AllMsgEvent>) {
        if !e.is_group() {
            return;
        }

        let text = match e.borrow_text() {
            Some(v) => v,
            None => return,
        };

        if text.trim() != self.config.my_times_cmd {
            return;
        }

        let group_id = e.group_id.unwrap();
        let user_id = e.user_id;
        let user_info_lock = self.user_info.lock().unwrap();

        if let Some(user_data) = user_info_lock.get(&user_id) {
            let group_times = user_data.group_total_times.get(&group_id).unwrap_or(&0);
            let total_times = user_data.total_times;
            let reply_msg = format!(
                "你在本群发送{}的次数为: {}\n你的总发送次数为: {}",
                self.name, group_times, total_times
            );
            e.reply(&reply_msg);
        } else {
            e.reply(&format!("你还没有发送过{}哦~", self.name));
        }
    }

    pub(crate) async fn process_images(
        &self,
        e: Arc<AllMsgEvent>,
        bot: Arc<RuntimeBot>,
        imgs_data: Vec<(Vec<u8>, ImageFormat)>,
        is_check_mode: bool,
    ) {
        if is_check_mode {
            self.send_with_img(e, bot, imgs_data).await;
        } else {
            self.send_not_img(e, bot, imgs_data).await;
        }
    }

    pub(crate) async fn send_with_img(
        &self,
        e: Arc<AllMsgEvent>,
        bot: Arc<RuntimeBot>,
        imgs_data: Vec<(Vec<u8>, ImageFormat)>,
    ) {
        let mut msg = Message::from(&self.config.reply_msg);
        let mut detected = false;
        let mut remove_img_path = Vec::new();

        let mut i = 0;
        for (img_data, img_type) in imgs_data {
            i += 1;
            let (res_img, prob) = match img_type {
                ImageFormat::Gif => {
                    let img = extract_frame_from_gif_bytes(&img_data, 0).await.unwrap();
                    match self.process_image_with_image(img) {
                        Ok(v) => v,
                        Err(err) => {
                            error!("{}", err);
                            return;
                        }
                    }
                }
                _ => {
                    let original_img = image::load_from_memory(&img_data).unwrap();
                    match self.process_image_with_image(original_img) {
                        Ok(v) => v,
                        Err(err) => {
                            error!("{}", err);
                            return;
                        }
                    }
                }
            };

            info!("{} prob: {}", self.name, prob);

            if prob >= self.config.trigger {
                detected = true;
                let filename = format!(
                    "{}-{}-output.png",
                    chrono::Local::now().format("%Y-%m-%d-%H-%M-%S"),
                    i
                );
                let output_path = self.data_path.join("tmp").join(filename);

                if let Some(parent_dir) = output_path.parent() {
                    if !parent_dir.exists() {
                        tokio::fs::create_dir_all(parent_dir).await.unwrap();
                    }
                }

                image::save_buffer(
                    &output_path,
                    &res_img,
                    res_img.width(),
                    res_img.height(),
                    image::ColorType::Rgba8,
                )
                .unwrap();

                if self.config.is_reply_trigger {
                    msg.push_text(format!("\n相似度：{:.2}", prob));
                }
                msg.push_image(output_path.to_str().unwrap());

                remove_img_path.push(output_path);
            }
        }

        if !detected {
            delete(remove_img_path).await;
            return;
        }

        e.reply_and_quote(msg);
        tokio::time::sleep(Duration::from_secs(1)).await;
        if self.config.is_delete_message {
            bot.delete_msg(e.message_id);
        }

        tokio::time::sleep(Duration::from_secs(10)).await;
        delete(remove_img_path).await;
    }

    pub(crate) async fn send_not_img(
        &self,
        e: Arc<AllMsgEvent>,
        bot: Arc<RuntimeBot>,
        imgs_data: Vec<(Vec<u8>, ImageFormat)>,
    ) {
        let mut msg = Message::from(&self.config.reply_msg);
        let mut is_detected = false;

        for (img_data, img_type) in imgs_data {
            let prob = match img_type {
                ImageFormat::Gif => {
                    let img = extract_frame_from_gif_bytes(&img_data, 0).await.unwrap();
                    match self.process_image(img) {
                        Ok(v) => v,
                        Err(err) => {
                            error!("{}", err);
                            return;
                        }
                    }
                }
                _ => {
                    let original_img = image::load_from_memory(&img_data).unwrap();
                    match self.process_image(original_img) {
                        Ok(v) => v,
                        Err(err) => {
                            error!("{}", err);
                            return;
                        }
                    }
                }
            };

            info!("{} prob: {}", self.name, prob);

            if prob >= self.config.trigger {
                is_detected = true;
                if self.config.is_reply_trigger {
                    msg.push_text(format!("\n相似度：{:.2}", prob));
                }
            }
        }

        if !is_detected {
            return;
        }

        let group_id = e.group_id.unwrap();
        let user_id = e.user_id;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        {
            let mut user_info_lock = self.user_info.lock().unwrap();
            let user_data = user_info_lock.entry(user_id).or_insert_with(|| UserInfo {
                total_times: 0,
                group_total_times: HashMap::new(),
                last_timestamp: HashMap::new(),
            });

            let last_timestamp = user_data.last_timestamp.get(&group_id).unwrap_or(&0);
            let time_diff = current_time - last_timestamp;

            if time_diff < self.config.ban_cooldown {
                bot.set_group_ban(group_id, user_id, self.config.ban_duration);
                e.reply(self.config.ban_msg.deref());
            }

            user_data.update_time(group_id, current_time);
        }

        e.reply_and_quote(msg);
        tokio::time::sleep(Duration::from_secs(1)).await;
        if self.config.is_delete_message {
            bot.delete_msg(e.message_id);
        }
    }

    pub(crate) fn process_image_with_image(
        &self,
        original_img: DynamicImage,
    ) -> ort::Result<(image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, f32)> {
        let (img_width, img_height) = (original_img.width(), original_img.height());
        let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);

        let mut input = Array::zeros((1, 3, 640, 640));
        for pixel in img.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2 .0;
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }

        let outputs: SessionOutputs = self.model.run(inputs!["images" => input.view()]?)?;
        let output = outputs["output0"]
            .try_extract_tensor::<f32>()?
            .t()
            .into_owned();

        let mut boxes = Vec::new();
        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if prob < 0.3 {
                continue;
            }

            let label = self.labels[class_id];
            let xc = row[0] / 640. * (img_width as f32);
            let yc = row[1] / 640. * (img_height as f32);
            let w = row[2] / 640. * (img_width as f32);
            let h = row[3] / 640. * (img_height as f32);
            boxes.push((
                BoundingBox {
                    x1: xc - w / 2.,
                    y1: yc - h / 2.,
                    x2: xc + w / 2.,
                    y2: yc + h / 2.,
                },
                label,
                prob,
            ));
        }

        boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
        let mut result = Vec::new();

        while !boxes.is_empty() {
            result.push(boxes[0]);
            boxes = boxes
                .iter()
                .filter(|box1| {
                    intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7
                })
                .copied()
                .collect();
        }

        let mut max_prob = 0.0;
        let mut dt = DrawTarget::new(img_width as _, img_height as _);

        for (bbox, label, _confidence) in result {
            if label == "xiong" {
                continue;
            }

            if _confidence > max_prob {
                max_prob = _confidence;
            }

            let mut pb = PathBuilder::new();
            pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
            let path = pb.finish();

            let color = SolidSource {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            };

            dt.stroke(
                &path,
                &Source::Solid(color),
                &StrokeStyle {
                    join: LineJoin::Round,
                    width: 4.,
                    ..StrokeStyle::default()
                },
                &DrawOptions::new(),
            );
        }

        let box_img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::RgbaImage::from_raw(
            img_width,
            img_height,
            dt.get_data()
                .iter()
                .flat_map(|&p| {
                    let a = (p >> 24) & 0xff;
                    let r = (p >> 16) & 0xff;
                    let g = (p >> 8) & 0xff;
                    let b = p & 0xff;
                    vec![r as u8, g as u8, b as u8, a as u8]
                })
                .collect(),
        )
        .unwrap();

        let mut res_img = image::RgbaImage::new(img_width, img_height);
        for (x, y, pixel) in res_img.enumerate_pixels_mut() {
            let original_pixel = original_img.get_pixel(x, y);
            let box_pixel = box_img.get_pixel(x, y);

            if box_pixel.0[3] > 0 {
                *pixel = image::Rgba([
                    box_pixel.0[0],
                    box_pixel.0[1],
                    box_pixel.0[2],
                    box_pixel.0[3],
                ]);
            } else {
                *pixel = image::Rgba([
                    original_pixel.0[0],
                    original_pixel.0[1],
                    original_pixel.0[2],
                    255,
                ]);
            }
        }

        Ok((res_img, max_prob))
    }

    pub(crate) fn process_image(&self, original_img: DynamicImage) -> ort::Result<f32> {
        let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);

        let mut input = Array::zeros((1, 3, 640, 640));
        for pixel in img.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2 .0;
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }

        let outputs: SessionOutputs = self.model.run(inputs!["images" => input.view()]?)?;
        let output = outputs["output0"]
            .try_extract_tensor::<f32>()?
            .t()
            .into_owned();

        let mut max_prob = 0.0;
        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if self.labels[class_id] == self.labels[0] && prob > max_prob {
                max_prob = prob;
            }
        }

        Ok(max_prob)
    }
}


pub(crate) fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub(crate) fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

pub(crate) async fn download_img(
    url: &str,
) -> Result<(Vec<u8>, ImageFormat), Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;
    if response.status().is_success() {
        let content = response.bytes().await?;
        let img_type = image::guess_format(&content)?;
        Ok((content.to_vec(), img_type))
    } else {
        Err("请求失败".into())
    }
}

pub(crate) async fn extract_frame_from_gif_bytes(
    data: &[u8],
    frame_index: usize,
) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    let cursor = Cursor::new(data);
    let decoder = GifDecoder::new(cursor)?;
    let frames = decoder.into_frames().collect_frames()?;

    if frame_index >= frames.len() {
        return Err("Frame index out of bounds".into());
    }

    let frame = frames[frame_index].clone();
    let dynamic_image = DynamicImage::ImageRgba8(frame.into_buffer());

    Ok(dynamic_image)
}

pub(crate) async fn delete(remove_img_path: Vec<PathBuf>) {
    for path in remove_img_path {
        if let Err(err) = tokio::fs::remove_file(&path).await {
            error!("{}", err);
            error!("path {}", path.to_str().unwrap());
        };
    }
}

use image::codecs::gif::GifDecoder;
use image::AnimationDecoder;
use image::{imageops::FilterType, GenericImageView};
use image::{DynamicImage, ImageFormat};
use kovi::bot::runtimebot::kovi_api::KoviApi as _;
use kovi::log::{error, info};
use kovi::utils::{load_json_data, save_json_data};
use kovi::{chrono, tokio, AllMsgEvent, Message, PluginBuilder as p, RuntimeBot};
use ndarray::{s, Array, Axis};
use ort::GraphOptimizationLevel;
use ort::{inputs, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Cursor;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

const LONG_MODEL: &[u8] = include_bytes!("../model/last.onnx");

static LABELS: [&str; 1] = ["nailong"];

#[derive(Clone, Serialize, Deserialize, Debug)]
struct UserInfo {
    total_times: u64,                     // 所有总次数
    group_total_times: HashMap<i64, u64>, // 本群总次数
    last_timestamp: HashMap<i64, u64>,
}
impl UserInfo {
    fn update_time(&mut self, group_id: i64, last_timestamp: u64) {
        // 更新总次数
        self.total_times += 1;

        // 更新本群总次数
        *self.group_total_times.entry(group_id).or_insert(0) += 1;

        // 更新最后时间戳
        self.last_timestamp.insert(group_id, last_timestamp);
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
struct Config {
    trigger: f32,
    start_cmd: String,
    start_msg: String,
    stop_cmd: String,
    stop_msg: String,
    reply_output_img_cmd: String,
    reply_msg: String,
    my_times_cmd: String,
    is_reply_trigger: bool,
    is_delete_message: bool,
    ban_cooldown: u64,
    ban_duration: usize,
    ban_msg: String,
}

#[kovi::plugin]
async fn main() {
    let bot = p::get_runtime_bot();
    let data_path = bot.get_data_path();
    let whitelist_path = bot.get_data_path().join("whitelist.json");
    let default_value = HashMap::<i64, bool>::new();
    let whitelist = load_json_data(default_value, &whitelist_path).unwrap();

    let whitelist = Arc::new(RwLock::new(whitelist));

    let user_info_path = Arc::new(data_path.join("user_info.json"));
    let user_info: Arc<Mutex<HashMap<i64, UserInfo>>> = Arc::new(Mutex::new(
        load_json_data(HashMap::new(), user_info_path.as_ref()).unwrap(),
    ));

    let defult_config = Config {
        trigger: 0.78,
        start_cmd: ".nailostart".to_string(),
        stop_cmd: ".nailostop".to_string(),
        start_msg: "喜欢发奶龙的小朋友你们好啊，📢📢📢，本群已开启奶龙戒严".to_string(),
        stop_msg: "📢📢📢，本群已关闭奶龙戒严".to_string(),
        reply_output_img_cmd: "检测".to_string(),
        reply_msg: "不准发奶龙哦，再发打你👊".to_string(),
        my_times_cmd: "我的奶龙".to_string(),
        is_reply_trigger: true,
        is_delete_message: true,
        ban_cooldown: 60,
        ban_duration: 60,
        ban_msg: "发发发发发，不准发了👊👊👊".to_string(),
    };

    let config = Arc::new(load_json_data(defult_config, data_path.join("config.json")).unwrap());

    // 加载 YOLO 模型
    let model = Arc::new(
        Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_memory(LONG_MODEL)
            .unwrap(),
    );
    info!("模型加载完毕");

    p::on_admin_msg({
        let config = config.clone();
        let whitelist = whitelist.clone();
        move |e| {
            let whitelist = whitelist.clone();
            let config = config.clone();
            async move {
                if e.text.is_none() {
                    return;
                }

                if (e.borrow_text().unwrap() != config.start_cmd)
                    && (e.borrow_text().unwrap() != config.stop_cmd)
                {
                    return;
                };

                if !e.is_group() {
                    return;
                };

                let mut whitelist_lock = whitelist.write().unwrap();


                if e.borrow_text().unwrap() == config.start_cmd {
                    whitelist_lock.insert(e.group_id.unwrap(), true);
                    e.reply(&config.start_msg);
                } else if e.borrow_text().unwrap() == config.stop_cmd {
                    whitelist_lock.insert(e.group_id.unwrap(), false);
                    e.reply(&config.stop_msg);
                } else {
                    return;
                }
            }
        }
    });

    p::drop({
        let whitelist = whitelist.clone();
        let whitelist_path = Arc::new(whitelist_path);
        let data_path = data_path.clone();
        let user_info = user_info.clone();
        let user_info_path = user_info_path.clone();
        move || {
            let whitelist = whitelist.clone();
            let whitelist_path = whitelist_path.clone();
            let data_path = data_path.clone();
            let user_info = user_info.clone();
            let user_info_path = user_info_path.clone();
            async move {
                {
                    let whitelist = whitelist.write().unwrap();
                    save_json_data(&*whitelist, whitelist_path.as_ref()).unwrap();
                }

                {
                    let user_info = user_info.lock().unwrap();
                    println!("{:?}", user_info);
                    save_json_data(&*user_info, user_info_path.as_ref()).unwrap();
                }

                // 删除 tmp 文件夹里面的临时图片
                let tmp_dir = data_path.join("tmp");
                if let Ok(mut entries) = tokio::fs::read_dir(&tmp_dir).await {
                    while let Some(entry) = entries.next_entry().await.unwrap_or(None) {
                        if let Ok(file_type) = entry.file_type().await {
                            if file_type.is_file() {
                                if let Err(e) = tokio::fs::remove_file(entry.path()).await {
                                    error!("Failed to delete file {:?}: {}", entry.path(), e);
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    p::on_group_msg({
        let config = config.clone();
        let user_info = user_info.clone();
        move |e| {
            let config = config.clone();
            let user_info = user_info.clone();
            async move {
                if e.group_id.is_none() {
                    return;
                }

                let text = match e.borrow_text() {
                    Some(v) => v,
                    None => return,
                };

                if text.trim() != config.my_times_cmd {
                    return;
                }

                let group_id = e.group_id.unwrap();
                let user_id = e.user_id;
                let user_info_lock = user_info.lock().unwrap();
                if let Some(user_data) = user_info_lock.get(&user_id) {
                    let group_times = user_data.group_total_times.get(&group_id).unwrap_or(&0);
                    let total_times = user_data.total_times;
                    let reply_msg = format!(
                        "你在本群发送奶龙的次数为: {}\n你的总发送次数为: {}",
                        group_times, total_times
                    );
                    e.reply(&reply_msg);
                } else {
                    e.reply("你还没有发送过奶龙哦~");
                }
            }
        }
    });

    p::on_group_msg({
        let model = model.clone();
        let data_path = Arc::new(data_path);
        let bot = bot.clone();
        let config = config.clone();
        move |e| {
            let model = model.clone();
            let bot = bot.clone();
            let data_path = data_path.clone();
            let config = config.clone();
            async move {
                match e.borrow_text() {
                    Some(v) => {
                        if v.trim() != config.reply_output_img_cmd {
                            return;
                        }
                    }

                    None => return,
                }

                let imgs = e.message.get("image");
                if imgs.is_empty() {
                    return;
                }
                let urls = imgs
                    .iter()
                    .map(|x| x.data.get("url").unwrap().as_str().unwrap())
                    .collect::<Vec<_>>();

                let mut imgs_data = Vec::new();
                for url in &urls {
                    match download_img(url).await {
                        Ok((data, format)) => imgs_data.push((data, format)),
                        Err(err) => {
                            error!("下载图片失败: {}", err);
                            continue;
                        }
                    }
                }

                if imgs_data.is_empty() {
                    info!("没有成功下载的图片");
                    return;
                }


                send_with_img(
                    model.clone(),
                    e,
                    bot.clone(),
                    imgs_data,
                    data_path.clone(),
                    config.clone(),
                )
                .await;
            }
        }
    });

    p::on_group_msg({
        let model = model.clone();
        let bot = bot.clone();
        let whitelist = whitelist.clone();
        let config = config.clone();
        let user_info = user_info.clone();
        move |e| {
            let model = model.clone();
            let whitelist = whitelist.clone();
            let bot = bot.clone();
            let config = config.clone();
            let user_info = user_info.clone();
            async move {
                if let Some(v) = e.borrow_text() {
                    if v.trim() == config.reply_output_img_cmd {
                        return;
                    }
                }

                {
                    let whitelist = whitelist.read().unwrap();
                    let group_id = e.group_id.as_ref().unwrap();
                    match whitelist.get(group_id) {
                        Some(v) => {
                            if !v {
                                return;
                            }
                        }
                        None => return,
                    }
                }

                let imgs = e.message.get("image");
                if imgs.is_empty() {
                    return;
                }
                let urls = imgs
                    .iter()
                    .map(|x| x.data.get("url").unwrap().as_str().unwrap())
                    .collect::<Vec<_>>();

                let mut imgs_data = Vec::new();
                for url in &urls {
                    match download_img(url).await {
                        Ok((data, format)) => imgs_data.push((data, format)),
                        Err(err) => {
                            error!("下载图片失败: {}", err);
                            continue;
                        }
                    }
                }

                if imgs_data.is_empty() {
                    info!("没有成功下载的图片");
                    return;
                }

                send_not_img(
                    model.clone(),
                    e.clone(),
                    bot.clone(),
                    imgs_data,
                    config.clone(),
                    user_info.clone(),
                )
                .await;
            }
        }
    });
}

async fn send_with_img(
    model: Arc<Session>,
    e: Arc<AllMsgEvent>,
    bot: Arc<RuntimeBot>,
    imgs_data: Vec<(Vec<u8>, ImageFormat)>,
    data_path: Arc<PathBuf>,
    config: Arc<Config>,
) {
    let mut msg = Message::from(&config.reply_msg);
    let mut nailong = false;
    let mut remove_img_path = Vec::new();

    let mut i = 0;
    for (img_data, img_type) in imgs_data {
        i += 1;
        let (res_img, prob) = match img_type {
            ImageFormat::Gif => {
                let img = extract_frame_from_gif_bytes(&img_data, 0).await.unwrap();
                match process_image_with_image(model.clone(), img) {
                    Ok(v) => v,
                    Err(err) => {
                        error!("{}", err);
                        return;
                    }
                }
            }
            _ => {
                let original_img = image::load_from_memory(&img_data).unwrap();
                match process_image_with_image(model.clone(), original_img) {
                    Ok(v) => v,
                    Err(err) => {
                        error!("{}", err);
                        return;
                    }
                }
            }
        };

        info!("nailong prob: {}", prob);

        if prob >= config.trigger {
            nailong = true;
            let filename = format!(
                "{}-{}-output.png",
                chrono::Local::now().format("%Y-%m-%d-%H-%M-%S"),
                i
            );
            let output_path = data_path.as_ref().join("tmp").join(filename);

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

            if config.is_reply_trigger {
                msg.push_text(format!("\n相似度：{:.2}", prob));
            }
            msg.push_image(output_path.to_str().unwrap());

            remove_img_path.push(output_path.clone());
        }
    }


    if !nailong {
        delete(remove_img_path).await;
        return;
    }
    e.reply_and_quote(msg);
    tokio::time::sleep(Duration::from_secs(1)).await;
    if config.is_delete_message {
        bot.delete_msg(e.message_id);
    }

    tokio::time::sleep(Duration::from_secs(10)).await;
    delete(remove_img_path).await;
}

async fn send_not_img(
    model: Arc<Session>,
    e: Arc<AllMsgEvent>,
    bot: Arc<RuntimeBot>,
    imgs_data: Vec<(Vec<u8>, ImageFormat)>,
    config: Arc<Config>,
    user_info: Arc<Mutex<HashMap<i64, UserInfo>>>,
) {
    let mut msg = Message::from(&config.reply_msg);
    let mut is_dragon = false;

    for (img_data, img_type) in imgs_data {
        let prob = match img_type {
            ImageFormat::Gif => {
                let img = extract_frame_from_gif_bytes(&img_data, 0).await.unwrap();
                match process_image(model.clone(), img) {
                    Ok(v) => v,
                    Err(err) => {
                        error!("{}", err);
                        return;
                    }
                }
            }
            _ => {
                let original_img = image::load_from_memory(&img_data).unwrap();
                match process_image(model.clone(), original_img) {
                    Ok(v) => v,
                    Err(err) => {
                        error!("{}", err);
                        return;
                    }
                }
            }
        };

        info!("nailong prob: {}", prob);


        if prob >= config.trigger {
            is_dragon = true;
            if config.is_reply_trigger {
                msg.push_text(format!("\n相似度：{:.2}", prob));
            }
        }
    }

    if !is_dragon {
        return;
    }


    let group_id = e.group_id.unwrap();
    let user_id = e.user_id;
    let current_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    {
        let mut user_info_lock = user_info.lock().unwrap();
        let user_data = user_info_lock.entry(user_id).or_insert_with(|| UserInfo {
            total_times: 0,
            group_total_times: HashMap::new(),
            last_timestamp: HashMap::new(),
        });


        let last_timestamp = user_data.last_timestamp.get(&group_id).unwrap_or(&0);
        let time_diff = current_time - last_timestamp;
        println!(
            "time:{}\n cur:{}\n last_:{}",
            time_diff, current_time, last_timestamp
        );
        if time_diff < config.ban_cooldown {
            // 禁言用户
            bot.set_group_ban(group_id, user_id, config.ban_duration);
            e.reply(config.ban_msg.deref());
        }

        user_data.update_time(group_id, current_time);
    }

    e.reply_and_quote(msg);
    tokio::time::sleep(Duration::from_secs(1)).await;
    if config.is_delete_message {
        bot.delete_msg(e.message_id);
    }
}

async fn delete(remove_img_path: Vec<PathBuf>) {
    for path in remove_img_path {
        if let Err(err) = tokio::fs::remove_file(&path).await {
            error!("{}", err);
            error!("path {}", path.to_str().unwrap());
        };
    }
}


fn process_image_with_image(
    model: Arc<Session>,
    original_img: DynamicImage,
) -> ort::Result<(image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, f32)> {
    let (img_width, img_height) = (original_img.width(), original_img.height());

    // 将图像调整为 640x640 大小
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);

    // 准备输入数据
    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        // 归一化像素值
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }


    // 运行 YOLO 推理
    let outputs: SessionOutputs = model.run(inputs!["images" => input.view()]?)?;
    let output = outputs["output0"]
        .try_extract_tensor::<f32>()?
        .t()
        .into_owned();

    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        // 找出概率最高的类别
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        // 过滤低概率检测结果
        if prob < 0.3 {
            continue;
        }
        let label = LABELS[class_id];
        // 计算边界框坐标
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

    // 按概率排序并应用非极大值抑制 (NMS)
    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    let mut result = Vec::new();

    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
            .copied()
            .collect();
    }


    let mut max_prob = 0.0;
    // 创建绘图目标
    let mut dt = DrawTarget::new(img_width as _, img_height as _);

    // 绘制边界框
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
        // 根据标签选择颜色
        let color = match label {
            "nailong" => SolidSource {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
            _ => SolidSource {
                r: 0x80,
                g: 0x10,
                b: 0x40,
                a: 0x80,
            },
        };
        // 绘制边界框
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

    // 将绘图结果转换为图像
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

    // 将box_img和原图进行合并
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

// 计算两个边界框的交集面积
fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

// 计算两个边界框的并集面积
fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

fn process_image(model: Arc<Session>, original_img: DynamicImage) -> ort::Result<f32> {
    // 将图像调整为 640x640 大小
    let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);

    // 准备输入数据
    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        // 归一化像素值
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    // 运行 YOLO 推理
    let outputs: SessionOutputs = model.run(inputs!["images" => input.view()]?)?;
    let output = outputs["output0"]
        .try_extract_tensor::<f32>()?
        .t()
        .into_owned();

    let mut max_loong_prob = 0.0;
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        // 找出概率最高的类别
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();

        if LABELS[class_id] == "nailong" && prob > max_loong_prob {
            max_loong_prob = prob;
        }
    }

    Ok(max_loong_prob)
}

async fn download_img(url: &str) -> Result<(Vec<u8>, ImageFormat), Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;
    if response.status().is_success() {
        let content = response.bytes().await?;
        let img_type = image::guess_format(&content)?;
        Ok((content.to_vec(), img_type))
    } else {
        Err("请求失败".into())
    }
}


async fn extract_frame_from_gif_bytes(
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

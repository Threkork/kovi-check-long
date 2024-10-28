use detector::{download_img, Detector};
use kovi::bot::runtimebot::kovi_api::KoviApi as _;
use kovi::log::error;
use kovi::utils::{load_json_data, save_json_data};
use kovi::{tokio, AllMsgEvent, PluginBuilder as p};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};


mod detector;

pub const LONG_MODEL: &[u8] = include_bytes!("../model/long.onnx");
pub const NAILONG_MODEL: &[u8] = include_bytes!("../model/nailong.onnx");

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

    // 龙图白名单
    let long_whitelist_path = bot.get_data_path().join("long_whitelist.json");
    let default_value = HashMap::<i64, bool>::new();
    let long_whitelist = load_json_data(default_value, &long_whitelist_path).unwrap();
    let long_whitelist = Arc::new(RwLock::new(long_whitelist));

    // 奶龙白名单
    let nailong_whitelist_path = bot.get_data_path().join("nailong_whitelist.json");
    let default_value = HashMap::<i64, bool>::new();
    let nailong_whitelist = load_json_data(default_value, &nailong_whitelist_path).unwrap();
    let nailong_whitelist = Arc::new(RwLock::new(nailong_whitelist));

    // 龙图用户信息
    let long_user_info_path = Arc::new(data_path.join("long_user_info.json"));
    let long_user_info: Arc<Mutex<HashMap<i64, UserInfo>>> = Arc::new(Mutex::new(
        load_json_data(HashMap::new(), long_user_info_path.as_ref()).unwrap(),
    ));

    // 奶龙用户信息
    let nailong_user_info_path = Arc::new(data_path.join("nailong_user_info.json"));
    let nailong_user_info: Arc<Mutex<HashMap<i64, UserInfo>>> = Arc::new(Mutex::new(
        load_json_data(HashMap::new(), nailong_user_info_path.as_ref()).unwrap(),
    ));

    // 龙图检测器配置
    let long_config = Config {
        trigger: 0.78,
        start_cmd: ".lostart".to_string(),
        stop_cmd: ".lostop".to_string(),
        start_msg: "喜欢发龙图的小朋友你们好啊，📢📢📢，本群已开启龙图戒严".to_string(),
        stop_msg: "📢📢📢，本群已关闭龙图戒严".to_string(),
        reply_output_img_cmd: "检测".to_string(),
        reply_msg: "不准发龙图哦，再发打你👊".to_string(),
        my_times_cmd: "我的龙图".to_string(),
        is_reply_trigger: true,
        is_delete_message: true,
        ban_cooldown: 60,
        ban_duration: 60,
        ban_msg: "发发发发发，不准发了👊👊👊".to_string(),
    };

    // 奶龙检测器配置
    let nailong_config = Config {
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

    let nailong_config =
        load_json_data(nailong_config, data_path.join("nailong_config.json")).unwrap();
    let long_config = load_json_data(long_config, data_path.join("long_config.json")).unwrap();

    // 创建检测器实例
    let long_detector = Detector::new(
        LONG_MODEL,
        long_config,
        vec!["loong", "xiong"],
        long_whitelist.clone(),
        long_user_info.clone(),
        data_path.clone(),
        "龙图".to_string(),
    );

    let nailong_detector = Detector::new(
        NAILONG_MODEL,
        nailong_config,
        vec!["nailong"],
        nailong_whitelist.clone(),
        nailong_user_info.clone(),
        data_path.clone(),
        "奶龙".to_string(),
    );


    let handle_admin = {
        let long_detector = long_detector.clone();
        let nailong_detector = nailong_detector.clone();
        move |e: Arc<AllMsgEvent>| {
            let long_detector = long_detector.clone();
            let nailong_detector = nailong_detector.clone();
            async move {
                long_detector.handle_admin_command(e.clone());
                nailong_detector.handle_admin_command(e);
            }
        }
    };

    let handle_my_times = {
        let long_detector = long_detector.clone();
        let nailong_detector = nailong_detector.clone();
        move |e: Arc<AllMsgEvent>| {
            let long_detector = long_detector.clone();
            let nailong_detector = nailong_detector.clone();
            async move {
                long_detector.handle_my_times(e.clone());
                nailong_detector.handle_my_times(e);
            }
        }
    };

    let handle_check = {
        let long_detector = long_detector.clone();
        let nailong_detector = nailong_detector.clone();
        let bot = bot.clone();
        move |e: Arc<AllMsgEvent>| {
            let long_detector = long_detector.clone();
            let nailong_detector = nailong_detector.clone();
            let bot = bot.clone();
            async move {
                if let Some(v) = e.borrow_text() {
                    if v.trim() != long_detector.config.reply_output_img_cmd {
                        return;
                    }
                } else {
                    return;
                }

                let imgs = e.message.get("image");
                if imgs.is_empty() {
                    return;
                }

                let urls: Vec<_> = imgs
                    .iter()
                    .map(|x| x.data.get("url").unwrap().as_str().unwrap())
                    .collect();

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
                    return;
                }

                long_detector
                    .process_images(e.clone(), bot.clone(), imgs_data.clone(), true)
                    .await;
                nailong_detector
                    .process_images(e, bot, imgs_data, true)
                    .await;
            }
        }
    };

    let handle_normal = {
        let long_detector = long_detector.clone();
        let nailong_detector = nailong_detector.clone();
        let bot = bot.clone();
        move |e: Arc<AllMsgEvent>| {
            let long_detector = long_detector.clone();
            let nailong_detector = nailong_detector.clone();
            let bot = bot.clone();
            async move {
                let group_id = if let Some(group_id) = e.group_id {
                    group_id
                } else {
                    return;
                };

                // Check whitelist
                let long_should_check = {
                    let whitelist = long_detector.whitelist.read().unwrap();
                    whitelist.get(&group_id).copied().unwrap_or(false)
                };

                let nailong_should_check = {
                    let whitelist = nailong_detector.whitelist.read().unwrap();
                    whitelist.get(&group_id).copied().unwrap_or(false)
                };

                if !long_should_check && !nailong_should_check {
                    return;
                }

                if let Some(v) = e.borrow_text() {
                    if v.trim() == long_detector.config.reply_output_img_cmd {
                        return;
                    }
                }

                let imgs = e.message.get("image");
                if imgs.is_empty() {
                    return;
                }

                let urls: Vec<_> = imgs
                    .iter()
                    .map(|x| x.data.get("url").unwrap().as_str().unwrap())
                    .collect();

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
                    return;
                }

                if long_should_check {
                    long_detector
                        .process_images(e.clone(), bot.clone(), imgs_data.clone(), false)
                        .await;
                }
                if nailong_should_check {
                    nailong_detector
                        .process_images(e, bot, imgs_data, false)
                        .await;
                }
            }
        }
    };

    // 注册处理器
    p::on_admin_msg(handle_admin);
    p::on_group_msg(handle_my_times);
    p::on_group_msg(handle_check);
    p::on_group_msg(handle_normal);

    // 注册插件卸载处理
    p::drop({
        let long_whitelist = long_whitelist.clone();
        let nailong_whitelist = nailong_whitelist.clone();
        let long_whitelist_path = Arc::new(long_whitelist_path);
        let nailong_whitelist_path = Arc::new(nailong_whitelist_path);
        let data_path = data_path.clone();
        let long_user_info = long_user_info.clone();
        let nailong_user_info = nailong_user_info.clone();
        let long_user_info_path = long_user_info_path.clone();
        let nailong_user_info_path = nailong_user_info_path.clone();
        move || {
            let long_whitelist = long_whitelist.clone();
            let nailong_whitelist = nailong_whitelist.clone();
            let long_whitelist_path = long_whitelist_path.clone();
            let nailong_whitelist_path = nailong_whitelist_path.clone();
            let data_path = data_path.clone();
            let long_user_info = long_user_info.clone();
            let nailong_user_info = nailong_user_info.clone();
            let long_user_info_path = long_user_info_path.clone();
            let nailong_user_info_path = nailong_user_info_path.clone();
            async move {
                {
                    let long_whitelist = long_whitelist.write().unwrap();
                    save_json_data(&*long_whitelist, long_whitelist_path.as_ref()).unwrap();
                }

                {
                    let nailong_whitelist = nailong_whitelist.write().unwrap();
                    save_json_data(&*nailong_whitelist, nailong_whitelist_path.as_ref()).unwrap();
                }

                {
                    let long_user_info = long_user_info.lock().unwrap();
                    println!("{:?}", long_user_info);
                    save_json_data(&*long_user_info, long_user_info_path.as_ref()).unwrap();
                }

                {
                    let nailong_user_info = nailong_user_info.lock().unwrap();
                    println!("{:?}", nailong_user_info);
                    save_json_data(&*nailong_user_info, nailong_user_info_path.as_ref()).unwrap();
                }

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
}

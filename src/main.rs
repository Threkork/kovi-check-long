use kovi::build_bot;

fn main() {
    println!("使用 Kovi 编写，项目链接: https://github.com/threkork/kovi");
    println!("仅支持正向ws链接，注意服务端配置");
    println!("检测龙图插件发送图片，需要和服务端在同一机子上面，不然发不了图片\n\n");

    build_bot!(check_alllong).run();
}

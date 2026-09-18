# 按任务运行

10.20.15.194 机器人的两个已成功放置任务，日常推理统一从 [固定包](../deployments/g2_194/README.md)
选择任务并生成配套命令。固定包不改变下述训练/数据处理目录。

共用转换和验证代码，每个任务保留自己的配置、episode 名单和记录。不为每批数据复制一套
转换器，也不把某次任务的路径、数量或成功条件写进公共代码。

- [折弯右臂放置 r0003](zhewan_right_place_r0003/README.md)
- [xichong 右臂放置 r0002](../examples/xichong_right_place_r0002/README.md)：历史任务记录；训练入口见 [主说明](../README.md)。
- [新右臂放置任务配置母本](../templates/right_place/README.md)

此前已成功的抓取、放置训练和真机记录仍保留在 `agibot/examples/`，机器人/GDK 差异仍记录
在 `agibot/docs/robots/`；不会随新任务覆盖。

本入口目前实现并验证的是右臂放置。左臂、双臂、抓取等任务可以复用目录形式，但应调整
字段、相机、modality、任务成功条件和测试，不能仅改一个名称就声称完成适配。

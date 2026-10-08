# 按任务运行

10.20.15.194 机器人的两个已成功放置任务，日常推理统一从 [放置固定包](../deployments/g2_194/README.md)
选择任务并生成配套命令；新抓取 r0002 使用自己的任务入口。推理部署不改变各任务的训练/数据处理目录。

共用转换和验证代码，每个任务保留自己的配置、episode 名单和记录。不为每批数据复制一套
转换器，也不把某次任务的路径、数量或成功条件写进公共代码。

- [折弯右臂放置 r0003](zhewan_right_place_r0003/README.md)
- [xichong 右臂放置 r0002](../examples/xichong_right_place_r0002/README.md)：历史任务记录；训练入口见 [主说明](../README.md)。
- [xichong 右臂抓取 r0002](g2_194/xichong_right_grasp_r0002/README.md)：10.20.15.194 机器人的独立数据、训练和推理记录。
- [新右臂放置任务配置母本](../templates/right_place/README.md)

此前已成功的抓取、放置训练和真机记录仍保留在 `agibot/examples/`，机器人/GDK 差异仍记录
在 `agibot/docs/robots/`；不会随新任务覆盖。

本入口目前实现并验证的是右臂抓取与放置。左臂、双臂或新的抓取/放置任务可以复用目录形式，但应调整
字段、相机、modality、任务成功条件和测试，不能仅改一个名称就声称完成适配。

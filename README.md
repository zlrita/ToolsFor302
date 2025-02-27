# ToolsFor302

some tools for 302

## 使用方法

### 1.把仓库拉到本地之后，把整个项目放在你的项目的目录下

|-Your project

|--Your Code

|--ToolsFor302

上述结构，不清楚就找我

### 2.在代码中直接import

添加`import ToolsFor302.Tools as t`  

可以按需添加，可能会分为Tools，dose_engine... 

(如果有的话就是 `import ToolsFor302.Dose_Engine` )

ps: as什么就随便了，不as也行

### 3.在代码中使用

举个栗子，你需要读取一个mhd格式的dose，

dose = t.load_mhd_dose(mhd_file_path)

这样就可以了，具体使用方法参考User_Manual

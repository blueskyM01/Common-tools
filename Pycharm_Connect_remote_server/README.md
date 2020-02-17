# Introduction
[//]: # (Image References)
[image1]: ./1.png
[image2]: ./2.png
[image3]: ./3.png
[image4]: ./4.png
[image5]: ./5.png
[image6]: ./6.png
[image7]: ./7.png
[image8]: ./8.png
[image8-1]: ./8-1.png
[image8-2]: ./8-2.png
[image9]: ./9.png
[image10]: ./10.png
[image10-1]: ./10-1.png
[image11]: ./11.png
`step1:`  
* Pycharm菜单栏，如下图所示，依次点击 Tools -> Deployment -> Configration  
![alt text][image1]  
* 点加号添加，选SFTP  
![alt text][image2]  
* 为服务器取个名称  
![alt text][image3]

* Connection下，协议选择sftp，接下来填写服务器主机IP，用户名，密码  
![alt text][image4]
* 连接成功  
![alt text][image5]

`step2:`  
* 在Mapping下，选择连接windows下的那部分代码和服务器上代码相连，本地Local path，服务器path，apply，OK，表示已经把本地的代码和服务器代码连接上了。  
![alt text][image6]

````
local path要求的是你填入本地的项目名称路径，接下来的是部署到服务器上的项目名称，这两个可以保持一致，也可以不保持一致。
当保持一致的时候，说明你只需要将当前的这一个项目做远程映射，即你只打算远程运行这一个项目，那么最后pychram会将这个项目上传到服务器你写的第二个路径的位置。
如下图所示：（自动 实时 同步）
````
![alt text][image7]

`step3:`  
* 光做好了远程映射还不行，这一步只是让你的pycharm能顺利找到文件，那么如何让pycharm告诉linux用什么去执行你的代码。还需要添加远程运行环境，从file-->settings。  
![alt text][image8]
![alt text][image8-1]
![alt text][image8-2]
![alt text][image9]
* 点“next”后出现下图

````
点击以后你会看到这样一个面板，这两个参数很关键，第一个参数是你要运行的python版本，比如我在linux 上安装了anaconda3。
第二个Sync folder是运行环境映射，表示这个使用的运行环境使用在哪个文件夹下，这里当然要填写我们第一步填写过的工程路径，这样就会自动将你本地的工程文件上传到箭头指向的远程位置。
````
![alt text][image10]
![alt text-1][image10-1]
`step4:`  
* 如果代码没有同步到服务器，可以点击这个同步，如下图所示。
![alt text][image11]








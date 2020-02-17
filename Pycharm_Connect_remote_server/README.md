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
[image9]: ./9.png
[image10]: ./10.png

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

* 然后填入你的linux  ip地址，和你登录linux的用户名称，然后点击next。接下来就会让你输入密码，当然可以使用putty的秘钥，这里直接使用密码，然后点击next。如下图所示：
![alt text][image9]
````
点击以后你会看到这样一个面板，这三个参数很关键，第一个参数是你要运行的python版本，比如我在linux 上安装了anaconda2、anaconda3。我在anaconda2中有python2.7版本、anaconda3中有python3.0和python3.7版本，linux系统自带的还有/usr/bin/python的默认版本。具体使用哪一个，请结合自己的项目选定。
第二个Sync folder是运行环境映射，表示这个使用的运行环境使用在哪个文件夹下，这里当然要填写我们第一步填写过的工程路径，这样就会自动将你本地的工程文件上传到箭头指向的远程位置，例如这里我就填写成/home/bxx-yll/mytest  (bxx-yll是我的用户名，mytest是我的项目名称)
````
![alt text][image10]








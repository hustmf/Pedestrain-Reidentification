# Pedestrain-Reidentification
A pedestrain reid system based on pytorch<br>
中文文档见pdf<br>
## Environment
windows10<br>
pyqt5.6.0+pytorch0.3.0+numpy（Anaconda4.2.0）<br>
wxpy pip install -U wxpy -i 'https://pypi.doubanio.com/simple/'<br>
language: python3.5<br>
It's a CPU version.<br>
### Application
Result is shown in “使用文档.pdf"<br>
We aim to realize PC & wechat & web appliction. Now we have finished PC and wechat version.<br>
Note: Only jpg format is supported now. But it's easy to modify code to make it adapt to different format<br>
###### PC Appliction
Run project.py<br>
Function:
1. Generate xml result for a whole query and reference set<br>
2. Open a picture, search the similarpictures in assigned folder, then rank and dispay top15 results on screen.<br>
Note:Query and Reference set is provided. You can also use new set. If you use new reference set, the features of this set have to be generated again and 'rfeatures.txt' will be generated in according folder.
This process may need some time related to your set size. And once you change any picture in a folder which txt is already generated. You have to delete the txt manually. It's a little bug in this version.<br>
###### Wechat Application
Run wxbot_new.py<br>
WeChat account A scans two-dimensional code displayed on the screen and assign account B as Interaction Object.<br>
A will send "Coming!"to B if it's ready.<br>
Note: code"my_friend = ensure_one(bot.friends().search(u'FZY'))" defines B. When you test, you should replace 'FZY' with B's wechat pet name.<br>
A and B should add as wechat friend before. Then B could send picture to A and then get 15 pictures from A.<br>

###### Finally, our model is trained according to https://github.com/Cysu/open-reid


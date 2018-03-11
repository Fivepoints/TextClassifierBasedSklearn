# TextClassifierBasedSklearn
### Use Sklearn toolkit for text classification
+ ###  1、data.txt 存储文本分词后向量,格式如下 ###
  <table>
      <tr>
          <td> 时光	飞逝	你	我	已	不	在	是	需要	父母	搀扶着	才能	行走	的	宝宝	现在	的	我们	任由	那双	并	不	强劲	的	脚 </td>
      </tr>
      <tr>
        <td> 来	独处	眺望	窗外	烈日炎炎	无处	可憩	只有	打开	电脑	寻找	能	让	自己	感兴趣	的	东西	网络	里	的	世界	真的	是	无穷无尽 </td>
      </tr>
      <tr>
        <td> 是	河南	人口数量	的	这	代表	着	河南	移动	已经	走进	了	河南	每	家庭	单位	移动	和	每个	人	的	信息	生活	息息相关 </td>
      </tr>
      <tr>
        <td> ... </td>
      </tr>
  </table>

+ ###  2、target.txt 存储类别标签,格式如下 ###
  <table>
      <tr>
          <td> 1 </td>
      </tr>
      <tr>
        <td> 0 </td>
      </tr>
      <tr>
        <td> 1 </td>
      </tr>
      <tr>
        <td> ... </td>
      </tr>
  </table>
+ ### 3、特征选择方法是卡方检验，使用Tf-idf来量化特征向量，特征数目为10000，总词典数量为12万+，最后准确率92.44%。

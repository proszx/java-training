# 写一个 bash 脚本以统计一个文本文件 words.txt 中每个单词出现的频率。

# 为了简单起见，你可以假设：

# words.txt只包括小写字母和 ' ' 。
# 每个单词只由小写字母组成。
# 单词间由一个或多个空格字符分隔。
# 示例:

# 假设 words.txt 内容如下：

# the day is sunny the the
# the sunny is is
# 你的脚本应当输出（以词频降序排列）：

# the 4
# is 3
# sunny 2
# day 1

cat words.txt |tr -s ' ' '\n'|sort|uniq -c|sort -r| awk '{print $2" "$1}'


# 给定一个包含电话号码列表（一行一个电话号码）的文本文件 file.txt，写一个 bash 脚本输出所有有效的电话号码。

# 你可以假设一个有效的电话号码必须满足以下两种格式： (xxx) xxx-xxxx 或 xxx-xxx-xxxx。（x 表示一个数字）

# 你也可以假设每行前后没有多余的空格字符。

# 示例:

# 假设 file.txt 内容如下：

# 987-123-4567
# 123 456 7890
# (123) 456-7890
# 你的脚本应当输出下列有效的电话号码：

# 987-123-4567
# (123) 456-7890
grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt
# 用户名	/^[a-z0-9_-]{3,16}$/
# 密码	/^[a-z0-9_-]{6,18}$/
# 十六进制值	/^#?([a-f0-9]{6}|[a-f0-9]{3})$/
# 电子邮箱	/^([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$/
# /^[a-z\d]+(\.[a-z\d]+)*@([\da-z](-[\da-z])?)+(\.{1,2}[a-z]+)+$/
# URL	/^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/
# IP 地址	/((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?)/
# /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/
# HTML 标签	/^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/
# 删除代码\\注释	(?<!http:|\S)//.*$
# Unicode编码中的汉字范围	/^[\u2E80-\u9FFF]+$/


# 给定一个文件 file.txt，转置它的内容。

# 你可以假设每行列数相同，并且每个字段由 ' ' 分隔.

# 示例:

# 假设 file.txt 文件内容如下：

# name age
# alice 21
# ryan 30
# 应当输出：

# name alice ryan
# age 21 30
awk '{
    for(i=1;i<=NF;i++){
        if(NR==1){
            res[i]=$i;
        }else{
            res[i]=res[i]" "$i
        }
    }
}
END{
    for(i=1;i<=NF;i++){
        print res[i]
    }
}' file.txt

# 给定一个文本文件 file.txt，请只打印这个文件中的第十行。

# 示例:

# 假设 file.txt 有如下内容：

# Line 1
# Line 2
# Line 3
# Line 4
# Line 5
# Line 6
# Line 7
# Line 8
# Line 9
# Line 10
# 你的脚本应当显示第十行：

# Line 10

awk 'NR==10' file.txt

# 编写一个 SQL 查询，来删除 Person 表中所有重复的电子邮箱，重复的邮箱里只保留 Id 最小 的那个。

# +----+------------------+
# | Id | Email            |
# +----+------------------+
# | 1  | john@example.com |
# | 2  | bob@example.com  |
# | 3  | john@example.com |
# +----+------------------+
# Id 是这个表的主键。
# 例如，在运行你的查询语句之后，上面的 Person 表应返回以下几行:

# +----+------------------+
# | Id | Email            |
# +----+------------------+
# | 1  | john@example.com |
# | 2  | bob@example.com  |
# +----+------------------+
#  

# 提示：

# 执行 SQL 之后，输出是整个 Person 表。
# 使用 delete 语句。

delete p1 from Person p1,Person p2
where p1.Email=p2.Email and p1.Id>p2.Id

# 给定一个 Weather 表，编写一个 SQL 查询，来查找与之前（昨天的）日期相比温度更高的所有日期的 Id。

# +---------+------------------+------------------+
# | Id(INT) | RecordDate(DATE) | Temperature(INT) |
# +---------+------------------+------------------+
# |       1 |       2015-01-01 |               10 |
# |       2 |       2015-01-02 |               25 |
# |       3 |       2015-01-03 |               20 |
# |       4 |       2015-01-04 |               30 |
# +---------+------------------+------------------+
# 例如，根据上述给定的 Weather 表格，返回如下 Id:

# +----+
# | Id |
# +----+
# |  2 |
# |  4 |
# +----+


select w1.Id
from 
Weather w1,
Weather w2
where 
datediff(w1.RecordDate,w2.RecordDate)=1
and w2.Temperature<w1.Temperature
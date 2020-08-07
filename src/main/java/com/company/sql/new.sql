
-- 表1: Person

-- +-------------+---------+
-- | 列名         | 类型     |
-- +-------------+---------+
-- | PersonId    | int     |
-- | FirstName   | varchar |
-- | LastName    | varchar |
-- +-------------+---------+
-- PersonId 是上表主键
-- 表2: Address

-- +-------------+---------+
-- | 列名         | 类型    |
-- +-------------+---------+
-- | AddressId   | int     |
-- | PersonId    | int     |
-- | City        | varchar |
-- | State       | varchar |
-- +-------------+---------+
-- AddressId 是上表主键
--  

-- 编写一个 SQL 查询，满足条件：无论 person 是否有地址信息，都需要基于上述两表提供 person 的以下信息：

--  

-- FirstName, LastName, City, State

SELECT * from Person p 
left join
Address a 
on p.PersonId=a.PersonId


-- 编写一个 SQL 查询，获取 Employee 表中第二高的薪水（Salary） 。

-- +----+--------+
-- | Id | Salary |
-- +----+--------+
-- | 1  | 100    |
-- | 2  | 200    |
-- | 3  | 300    |
-- +----+--------+

SELECT IFNULL((
    select distinct(salary)
    from Employee
    order by salary desc
    limit 1,1),null) 
as SecondHighestSalary

-- 编写一个 SQL 查询，获取 Employee 表中第 n 高的薪水（Salary）。

-- +----+--------+
-- | Id | Salary |
-- +----+--------+
-- | 1  | 100    |
-- | 2  | 200    |
-- | 3  | 300    |
-- +----+--------+
-- 例如上述 Employee 表，n = 2 时，应返回第二高的薪水 200。如果不存在第 n 高的薪水，那么查询应返回 null。

-- +------------------------+
-- | getNthHighestSalary(2) |
-- +------------------------+
-- | 200                    |
-- +------------------------+

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    SET N=N-1;
  RETURN (
      # Write your MySQL query statement below.
      SELECT DISTINCT salary FROM Employee  order by
      salary DESC LIMIT N,1
  );
END

-- 编写一个 SQL 查询来实现分数排名。

-- 如果两个分数相同，则两个分数排名（Rank）相同。请注意，平分后的下一个名次应该是下一个连续的整数值。换句话说，名次之间不应该有“间隔”。

-- +----+-------+
-- | Id | Score |
-- +----+-------+
-- | 1  | 3.50  |
-- | 2  | 3.65  |
-- | 3  | 4.00  |
-- | 4  | 3.85  |
-- | 5  | 4.00  |
-- | 6  | 3.65  |
-- +----+-------+
-- 例如，根据上述给定的 Scores 表，你的查询应该返回（按分数从高到低排列）：

-- +-------+------+
-- | Score | Rank |
-- +-------+------+
-- | 4.00  | 1    |
-- | 4.00  | 1    |
-- | 3.85  | 2    |
-- | 3.65  | 3    |
-- | 3.65  | 3    |
-- | 3.50  | 4    |
-- +-------+------+
-- 重要提示：对于 MySQL 解决方案，如果要转义用作列名的保留字，可以在关键字之前和之后使用撇号。例如 `Rank`

select s1.Score, count(distinct(s2.Score)) 'Rank'
from Scores s1, Scores s2
where s1.Score<=s2.Score
group by s1.Id
order by s1.Score DESC;

-- 编写一个 SQL 查询，查找所有至少连续出现三次的数字。

-- +----+-----+
-- | Id | Num |
-- +----+-----+
-- | 1  |  1  |
-- | 2  |  1  |
-- | 3  |  1  |
-- | 4  |  2  |
-- | 5  |  1  |
-- | 6  |  2  |
-- | 7  |  2  |
-- +----+-----+
-- 例如，给定上面的 Logs 表， 1 是唯一连续出现至少三次的数字。

-- +-----------------+
-- | ConsecutiveNums |
-- +-----------------+
-- | 1               |
-- +-----------------+

select distinct a.Num as ConsecutiveNums 
from Logs as a, Logs as b,Logs as c
where a.num = b.num and b.num = c.num
and a.id= b.id-1 and b.id= c.id-1


-- Employee 表包含所有员工，他们的经理也属于员工。每个员工都有一个 Id，此外还有一列对应员工的经理的 Id。

-- +----+-------+--------+-----------+
-- | Id | Name  | Salary | ManagerId |
-- +----+-------+--------+-----------+
-- | 1  | Joe   | 70000  | 3         |
-- | 2  | Henry | 80000  | 4         |
-- | 3  | Sam   | 60000  | NULL      |
-- | 4  | Max   | 90000  | NULL      |
-- +----+-------+--------+-----------+
-- 给定 Employee 表，编写一个 SQL 查询，该查询可以获取收入超过他们经理的员工的姓名。在上面的表格中，Joe 是唯一一个收入超过他的经理的员工。

-- +----------+
-- | Employee |
-- +----------+
-- | Joe      |
-- +----------+

select e1.Name as Employee 
from Employee as e1, Employee as e2 
where e1.ManagerId = e2.Id
And e1.salary>e2.salary

-- 编写一个 SQL 查询，查找 Person 表中所有重复的电子邮箱。

-- 示例：

-- +----+---------+
-- | Id | Email   |
-- +----+---------+
-- | 1  | a@b.com |
-- | 2  | c@d.com |
-- | 3  | a@b.com |
-- +----+---------+
-- 根据以上输入，你的查询应返回以下结果：

-- +---------+
-- | Email   |
-- +---------+
-- | a@b.com |
-- +---------+
-- 说明：所有电子邮箱都是小写字母。

select distinct a.email from 
Person a,Person b 
where a.Email = b.Email and a.Id!=b.Id


-- 某网站包含两个表，Customers 表和 Orders 表。编写一个 SQL 查询，找出所有从不订购任何东西的客户。

-- Customers 表：

-- +----+-------+
-- | Id | Name  |
-- +----+-------+
-- | 1  | Joe   |
-- | 2  | Henry |
-- | 3  | Sam   |
-- | 4  | Max   |
-- +----+-------+
-- Orders 表：

-- +----+------------+
-- | Id | CustomerId |
-- +----+------------+
-- | 1  | 3          |
-- | 2  | 1          |
-- +----+------------+
-- 例如给定上述表格，你的查询应返回：

-- +-----------+
-- | Customers |
-- +-----------+
-- | Henry     |
-- | Max       |
-- +-----------+

select c.Name as Customers 
from Customers c 
left join Orders o on o.CustomerId = c.Id 
where o.Id is null;

-- Employee 表包含所有员工信息，每个员工有其对应的 Id, salary 和 department Id。

-- +----+-------+--------+--------------+
-- | Id | Name  | Salary | DepartmentId |
-- +----+-------+--------+--------------+
-- | 1  | Joe   | 70000  | 1            |
-- | 2  | Henry | 80000  | 2            |
-- | 3  | Sam   | 60000  | 2            |
-- | 4  | Max   | 90000  | 1            |
-- +----+-------+--------+--------------+
-- Department 表包含公司所有部门的信息。

-- +----+----------+
-- | Id | Name     |
-- +----+----------+
-- | 1  | IT       |
-- | 2  | Sales    |
-- +----+----------+
-- 编写一个 SQL 查询，找出每个部门工资最高的员工。例如，根据上述给定的表格，Max 在 IT 部门有最高工资，Henry 在 Sales 部门有最高工资。

-- +------------+----------+--------+
-- | Department | Employee | Salary |
-- +------------+----------+--------+
-- | IT         | Max      | 90000  |
-- | Sales      | Henry    | 80000  |
-- +------------+----------+--------+

select
    d.Name as Department,
    e.Name as Employee,
    e.Salary as Salary
from Employee e,Department d
where e.DepartmentId=d.Id
    and (e.Salary ,e.DepartmentId) 
    in (select max(Salary),DepartMentId from Employee GROUP BY DepartmentId);


-- Employee 表包含所有员工信息，每个员工有其对应的工号 Id，姓名 Name，工资 Salary 和部门编号 DepartmentId 。

-- +----+-------+--------+--------------+
-- | Id | Name  | Salary | DepartmentId |
-- +----+-------+--------+--------------+
-- | 1  | Joe   | 85000  | 1            |
-- | 2  | Henry | 80000  | 2            |
-- | 3  | Sam   | 60000  | 2            |
-- | 4  | Max   | 90000  | 1            |
-- | 5  | Janet | 69000  | 1            |
-- | 6  | Randy | 85000  | 1            |
-- | 7  | Will  | 70000  | 1            |
-- +----+-------+--------+--------------+
-- Department 表包含公司所有部门的信息。

-- +----+----------+
-- | Id | Name     |
-- +----+----------+
-- | 1  | IT       |
-- | 2  | Sales    |
-- +----+----------+
-- 编写一个 SQL 查询，找出每个部门获得前三高工资的所有员工。例如，根据上述给定的表，查询结果应返回：

-- +------------+----------+--------+
-- | Department | Employee | Salary |
-- +------------+----------+--------+
-- | IT         | Max      | 90000  |
-- | IT         | Randy    | 85000  |
-- | IT         | Joe      | 85000  |
-- | IT         | Will     | 70000  |
-- | Sales      | Henry    | 80000  |
-- | Sales      | Sam      | 60000 
select
    d.Name as Department,
    e.Name as Employee,
    e.Salary as Salary
from Employee as e left join Department as d
on e.DepartmentId=d.Id
where e.Id in (
    select e1.Id from Employee  as e1 
    left join Employee as e2 
    on e1.DepartmentId=e2.DepartmentId and e1.Salary<e2.Salary
    group by e1.Id
    having count(distinct e2.Salary)<=2
)
and e.DepartmentId in (select Id from Department)
order by d.Id asc,e.Salary desc

-- 另外一种写法
select P2.Name as Department,P3.Name as Employee,P3.Salary as Salary
FROM Employee as P3
INNER JOIN Department AS P2
ON P2.ID=P3.DepartmentId
Where(
    SELECT Count(Distinct Salary)
    FROM Employee as P4
    WHERE P3.DepartmentId=P4.DepartmentId
    AND P4.Salary >=P3.Salary
)<=3
order by DepartmentId,Salary DESC


-- Trips 表中存所有出租车的行程信息。每段行程有唯一键 Id，Client_Id 和 Driver_Id 是 Users 表中 Users_Id 的外键。
-- Status 是枚举类型，枚举成员为 (‘completed’, ‘cancelled_by_driver’, ‘cancelled_by_client’)。
-- +----+-----------+-----------+---------+--------------------+----------+
-- | Id | Client_Id | Driver_Id | City_Id |        Status      |Request_at|
-- +----+-----------+-----------+---------+--------------------+----------+
-- | 1  |     1     |    10     |    1    |     completed      |2013-10-01|
-- | 2  |     2     |    11     |    1    | cancelled_by_driver|2013-10-01|
-- | 3  |     3     |    12     |    6    |     completed      |2013-10-01|
-- | 4  |     4     |    13     |    6    | cancelled_by_client|2013-10-01|
-- | 5  |     1     |    10     |    1    |     completed      |2013-10-02|
-- | 6  |     2     |    11     |    6    |     completed      |2013-10-02|
-- | 7  |     3     |    12     |    6    |     completed      |2013-10-02|
-- | 8  |     2     |    12     |    12   |     completed      |2013-10-03|
-- | 9  |     3     |    10     |    12   |     completed      |2013-10-03| 
-- | 10 |     4     |    13     |    12   | cancelled_by_driver|2013-10-03|
-- +----+-----------+-----------+---------+--------------------+----------+
-- Users 表存所有用户。每个用户有唯一键 Users_Id。Banned 表示这个用户是否被禁止，
-- Role 则是一个表示（‘client’, ‘driver’, ‘partner’）的枚举类型。
-- +----------+--------+--------+
-- | Users_Id | Banned |  Role  |
-- +----------+--------+--------+
-- |    1     |   No   | client |
-- |    2     |   Yes  | client |
-- |    3     |   No   | client |
-- |    4     |   No   | client |
-- |    10    |   No   | driver |
-- |    11    |   No   | driver |
-- |    12    |   No   | driver |
-- |    13    |   No   | driver |
-- +----------+--------+--------+
-- 写一段 SQL 语句查出 2013年10月1日 至 2013年10月3日 期间非禁止用户的取消率。基于上表，你的 SQL 语句应返回如下结果，取消率（Cancellation Rate）保留两位小数。
-- 取消率的计算方式如下：(被司机或乘客取消的非禁止用户生成的订单数量) / (非禁止用户生成的订单总数)
-- +------------+-------------------+
-- |     Day    | Cancellation Rate |
-- +------------+-------------------+
-- | 2013-10-01 |       0.33        |
-- | 2013-10-02 |       0.00        |
-- | 2013-10-03 |       0.50        |
-- +------------+-------------------+

select 
    t.request_at as Day,(
        round(count(if(status !='completed',status,null))/count(status),2)
    ) as 'Cancellation Rate'
from User u inner join Trips t
on u.Users_Id = t.Client_Id
and  u.banned != 'Yes'
where t.request_at>= '2013-10-01'
and t.request_at<='2013-10-03'
group by
    t.request_at
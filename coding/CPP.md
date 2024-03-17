
# C++ 编程基础

[toc]

## 常用

### 1. 智能指针

自动管理内存，在离开作用域时释放内存，避免

- std::unique_ptr : **独占所有权的智能指针**，确保同一时刻只有一个指针可以拥有和访问资源，当其被销毁或者重制时，会自动释放资源。适用于管理单个对象（pass）或动态分配的数组。

`make_unique`是一个函数模板，用于创建并返回一个`std::unique_ptr`智能指针
    
```cpp
std::unique_ptr<OperationDefinition> def =
	std::make_unique<OperationDefinition>(op, nameLoc, endLoc);
```
    
- std::shared_ptr: 共享所有权的智能指针，允许多个指针共享同一个资源，使用引用计数来跟踪资源的所有权，当最后一个shared_ptr被销毁或重制时，资源才会释放。适用于跨多个对象共享资源

- std::weak_ptr: 作为std::shared_ptr的辅助类，允许观察和访问由std::shared_ptr管理的资源，但不会增加引用计数。用于解决std::share_ptr造成的循环引用，使用其允许你创建一个指向由`std::shared_ptr`管理的资源的非拥有（弱）引用，而不会增加引用计数。它通过解除`std::shared_ptr`的循环引用来避免内存泄漏。

### 2. lambda编程

```cpp
// [] : 捕获列表，可以是值捕获、引用捕获或者不捕获任何变量
[capture clause](parameters) -> return_type {
    // Lambda函数体
    // 可以访问外部变量，参数等
    return expression; // 可选
};
```

用 `[&]` 可以捕获外面的值，如果lambda函数内使用外面的值较少，可以直接加在 `[]` 内

最好指定输出格式

```cpp
auto getReassociations = [&](const DenseSet<int64_t>& dimIndexSet) -> SmallVector<ReassociationIndices> {
auto getNewPermutation = [](const SmallVector<int64_t>& relativeOrder) -> SmallVector<int64_t> {
```

```cpp
llvm::for_each(relativeOrder, [](int64_t i) {llvm::outs() << i << " ";});

llvm::all_of(llvm::zip(array, array.slice(1)), [](const auto& pair) {
   return std::get<0>(pair) <= std::get<1>(pair);
});

llvm::find_if(shapeIndexs, [&](int64_t shapeIndex) {
   return !oneSizeDimIndexsSet.count(shapeIndex); 
});
```
### 3. 关键字

- const
const 表示常量。在成员函数声明和定义中，const 关键字表示该函数是一个常量成员函数，即**该函数不会修改对象的成员变量**。

- final
final 用于**防止类被继承**。当类被声明为 final，表示该类不能被其他类继承。

- final
public 是 C++ 中的访问修饰符之一，用于指定类的成员的访问权限。public 成员可以被任何类或函数访问。表示继承的成员和方法在派生类中是公共的，可以被外部访问。

- override
override 是 C++11 引入的关键字，用于**显式地声明一个成员函数覆盖了基类的虚函数**。在这里，这样做有助于提高代码的可读性和可维护性，同时也可以帮助编译器检查函数是否正确地覆盖了基类的虚函数。

## **1. 内存分区模型**

C++程序在执行时，将内存大方向划分为**4个区域**

- 代码区：存放函数体的二进制代码，由操作系统进行管理的
- 全局区：存放全局变量和静态变量static以及常量，程序作用后释放
- 栈区：由编译器自动分配释放, 存放函数的参数值,局部变量等 **(函数作用后释放)**
- 堆区：由程序员分配new和释放,若程序员不释放,程序结束时由操作系统回收 **(程序周期结束后释放)**

### **1.1 程序运行前**

在程序编译后，生成了**exe可执行程序**，未执行该程序前可分为两个区域

- **代码区：**
    
    存放CPU执行的机器指令
    
    代码区是**共享**的，共享的目的是对于频繁被执行的程序，只需要内存中有一份代码即可
    
    代码区是**只读**的，使其只读的原因是防止程序意外地修改了它的指令
    
- **全局区：**
    
    全局变量和静态变量(static)存放于此
    
    全局区还包含了常量区(const)，字符串常量和其他常量也存放于此
    
    该区域的数据在程序结束后由操作系统释放
    

### **1.2 程序运行后**

- **栈区：**
    
    由编译器自动分配释放，存放函数的参数值、局部变量等
    
    注意：局部变量在函数作用结束后就会被释放，所以返回局部变量的地址没有意义
    
- **堆区：**
    
    由程序员分配、释放，若程序员不是放，程序结束时由操作系统回收
    
    在C++中主要利用new在堆区开辟内存
    

> 局部变量、局部常量存放在栈区
> 
> 
> 全局变量和静态变量、全局常量存放在全局区
> 

## **2. 左右值**

### **2.1 概念**

左值: 表示可以在等号的左侧出现的表达式。左值是一块可以被识别的内存,所以我们可以获取其地址。典型的左值有:变量、数组元素、指针等。例子:

```cpp
int x = 10;  // x是左值
int* p = &x;   // 可以获取x的地址
```

右值: 表示只能在等号的右侧使用的表达式。右值不是一块可以被识别的内存,所以我们不能获取其地址。典型的右值有:字面量、运算产生的临时对象等。例子:

```cpp
10;             // 10是右值,不能获取其地址
x + y;          // 运算产生的临时对象是右值
```

- 左值是可寻址(有地址)的变量，具有永久性，数据绑定；
- 右值一般是不可寻址的常量，或在**表达式求值过程**中创建的无名临时变量，短暂性，让临时变量不消失，直接将目标指向临时变量，避免无意义的复制，减缓内存开销。

> 左值用于写操作，可以存储数据；
右值用于读操作，读到的数据放在一个看不见的临时变量
> 

**区别**：

- 地址:左值有地址,右值没有地址
- 生命周期:左值在程序的多个位置可使用,右值表达式计算完成就消失
- 赋值操作:左值可作为赋值操作的左操作数,右值只能作为赋值操作的右操作数。即左值可以被修改，而右值不能。

### **2.2 左值引用和右值引用**

引用的本质是指针常量 `int &y=x` 等价于 `int* const y = &a;`

- 左值引用：引用一个对象；
- 右值引用：C++中右值引用可以实现“移动语义”，通过&&获得右值引用
    
    ```cpp
    int x = 6; // x是左值，6是右值
    int &y = x; // 左值引用，y引用x
    
    int &z1 = x * 6; // 错误，x*6是一个右值
    const int &z2 =  x * 6; // 正确，可以将一个const引用绑定到一个右值
    
    int &&z3 = x * 6; // 正确，右值引用
    int &&z4 = x; // 错误，x是一个左值
    ```
    

### **2.3 左右值引用示例**

引用的注意事项：①引用必须初始化；②引用初始化后不可以改变

```cpp
int main() {

	int a = 10;
	int b = 20;
	//int &c; //错误，引用必须初始化
	int &c = a; //一旦初始化后，就不可以更改
	c = b; //这是赋值操作，不是更改引用

	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
	cout << "c = " << c << endl;

	return 0;
}
```

上述代码中

`int &c = a;`就是左值引用(赋地址操作)，c的地址绑定了a，**c和a会一起一起改变**

`c = b;`是右值引用(赋值操作),c得到了b地址中存储的值，同时a也跟着改变

因此输出为

> a = 20
b = 20
c = 20
> 

## **3. 类和对象**

C++面向对象的三大特性为：封装、多态、继承

对于C++来说，万事万物皆为对象，对象上有其属性和行为

### **3.1 封装**

#### **3.1.1 封装的意义**

- **封装意义一：**

在设计类的时候，属性和行为写在一起，表现事物

**语法：** `class 类名{   访问权限： 属性  / 行为  };`

- **封装意义二：**

类在设计时，可以把属性和行为放在不同的权限下，加以控制

访问权限有三种：

| 权限标识 | 权限名称 | 类内 | 类外 |
| --- | --- | --- | --- |
| public | 公共权限 | 类内可以访问 | 类外可以访问 |
| protected | 保护权限 | 类内可以访问 | 类外不可以访问 |
| private | 私有权限 | 类内可以访问 | 类外不可以访问 |

#### **3.1.2 struct和class区别**

唯一区别： 默认访问权限不同

- struct 默认权限为公共
- class 默认权限为私有

#### **3.1.3 成员属性设置为私有**

优点1：将所有成员属性设置为私有，可以自己控制读写权限

优点2：对于写权限，我们可以检测数据的有效性

因此在封装时，若使用`class`

```cpp
class student
{
private:  // 默认
    string name;
    int age;
    int stu_id[10];
}
```

若使用`struct`

```cpp
namespace // 将结构体对外隐藏
{
    template <typename T>
    struct student
    {
        string name;
        int age;
        int stu_id[10];
    }
}
```

这样能保证封装内容的独立性与完整性

### **3.2 对象的初始化和清理**

每个对象在生成时会进行初始设置，在销毁前会有清除数据的设置。

#### **3.2.1 构造函数和析构函数**

c++提供了**构造函数**和**析构函数**完成上述任务，这两个函数将会被编译器自动调用，完成对象初始化和清理工作。

对象的初始化和清理工作是编译器强制要我们做的事情，如果**我们不提供构造和析构，编译器会提供**

**编译器提供的构造函数和析构函数是空实现。**

- 构造函数：主要作用在于创建对象时为对象的成员属性赋值，构造函数由编译器自动调用，无须手动调用。
- 析构函数：主要作用在于对象**销毁前**系统自动调用，执行一些清理工作。

**构造函数语法：**`类名(){}`

1. 构造函数，没有返回值也不写void
2. 函数名称与类名相同
3. 构造函数可以有参数，因此可以发生重载
4. 程序在调用对象时候会自动调用构造，无须手动调用,而且只会调用一次

**析构函数语法：** `~类名(){}`

1. 析构函数，没有返回值也不写void
2. 函数名称与类名相同,在名称前加上符号 ~
3. 析构函数不可以有参数，因此不可以发生重载
4. 程序在对象销毁前会自动调用析构，无须手动调用,而且只会调用一次

#### **3.2.2 构造函数的分类和调用**

两种分类方式：

按参数分为： 有参构造和无参构造

按类型分为： 普通构造和拷贝构造

三种调用方式：

括号法

显示法

隐示转换法

```cpp
//1、构造函数分类
// 按照参数分类分为 有参和无参构造   无参又称为默认构造函数
// 按照类型分类分为 普通构造和拷贝构造

class Person {
public:
	//无参（默认）构造函数
	Person() {
		cout << "无参构造函数!" << endl;
	}
	//有参构造函数
	Person(int a) {
		age = a;
		cout << "有参构造函数!" << endl;
	}
	//拷贝构造函数
	Person(const Person& p) {
		age = p.age;
		cout << "拷贝构造函数!" << endl;
	}
	//析构函数
	~Person() {
		cout << "析构函数!" << endl;
	}
public:
	int age;
};

//2、构造函数的调用
//调用无参构造函数
void test01() {
	Person p; //调用无参构造函数
}

//调用有参的构造函数
void test02() {

	//2.1  括号法，常用
	Person p1(10);
	//注意1：调用无参构造函数不能加括号，如果加了编译器认为这是一个函数声明
	//Person p2();

	//2.2 显式法
	Person p2 = Person(10);
	Person p3 = Person(p2);
	//Person(10)单独写就是匿名对象  当前行结束之后，马上析构

	//2.3 隐式转换法
	Person p4 = 10; // Person p4 = Person(10);
	Person p5 = p4; // Person p5 = Person(p4);

	//注意2：不能利用 拷贝构造函数 初始化匿名对象 编译器认为是对象声明
	//Person p5(p4);
}

int main()
{

	test01();
	//test02();

	return 0;
}
```

#### **3.2.3 拷贝函数**

默认情况下，C++便提起至少给一个类添加三个函数

1. 默认构造函数（无参，函数体为空）
2. 默认析构函数（无参，函数体为空）
3. 默认拷贝构造函数，对属性进行值拷贝
- **浅拷贝：简单的复制拷贝操作**
- **深拷贝：在堆区重新申请空间，进行拷贝操作**

如果属性有在堆区开辟的，一定要自己提供拷贝构造函数，防止浅拷贝带来的问题

```cpp
class Person {
public:
	//无参（默认）构造函数
	Person() {
		cout << "无参构造函数!" << endl;
	}
	//有参构造函数
	Person(int age ,int height) {

		cout << "有参构造函数!" << endl;

		m_age = age;
		m_height = new int(height);

	}
	//拷贝构造函数
	Person(const Person& p) {
		cout << "拷贝构造函数!" << endl;
		//如果不利用深拷贝在堆区创建新内存，会导致浅拷贝带来的重复释放堆区问题
		m_age = p.m_age;
		m_height = new int(*p.m_height);

	}

	//析构函数
	~Person() {
		cout << "析构函数!" << endl;
		if (m_height != NULL)
		{
			delete m_height;
		}
	}
public:
	int m_age;
	int* m_height;
};

void test01()
{
	Person p1(18, 180);

	Person p2(p1);

	cout << "p1的年龄： " << p1.m_age << " 身高： " << *p1.m_height << endl;

	cout << "p2的年龄： " << p2.m_age << " 身高： " << *p2.m_height << endl;
}

int main()
{
	test01();
	return 0;
}
```

#### **3.2.4 初始化列表(初始化方法)**

**语法：**`构造函数()：属性1(值1),属性2（值2）... {}`

```cpp
class Person {
public:

	////传统方式初始化
	//Person(int a, int b, int c) {
	//	m_A = a;
	//	m_B = b;
	//	m_C = c;
	//}

	//初始化列表方式初始化
	Person(int a, int b, int c) :m_A(a), m_B(b), m_C(c) {}

private:
	int m_A;
	int m_B;
	int m_C;
};
```

### **3.3 继承**

有些类与类之间存在特殊的关系，例如下图中：

我们发现，定义这些类时，下级别的成员除了拥有上一级的共性，还有自己的特性。

这个时候我们就可以考虑利用继承的技术，减少重复代码

#### **3.3.1 继承的基本概念**

1. 继承的语法
    
    `class 子类 : 继承方式  父类`
    
2. **继承方式一共有三种：**
- 公共继承
- 保护继承
- 私有继承

```cpp
class Base1
{
public:
	int m_A;
protected:
	int m_B;
private:
	int m_C;
};

//公共继承
class Son1 :public Base1
{
public:
	void func()
	{
		m_A; //可访问 public权限
		m_B; //可访问 protected权限
		//m_C; //不可访问
	}
};

void myClass()
{
	Son1 s1;
	s1.m_A; //其他类只能访问到公共权限
}

//保护继承
class Base2
{
public:
	int m_A;
protected:
	int m_B;
private:
	int m_C;
};
class Son2:protected Base2
{
public:
	void func()
	{
		m_A; //可访问 protected权限
		m_B; //可访问 protected权限
		//m_C; //不可访问
	}
};
void myClass2()
{
	Son2 s;
	//s.m_A; //不可访问
}

//私有继承
class Base3
{
public:
	int m_A;
protected:
	int m_B;
private:
	int m_C;
};
class Son3:private Base3
{
public:
	void func()
	{
		m_A; //可访问 private权限
		m_B; //可访问 private权限
		//m_C; //不可访问
	}
};
class GrandSon3 :public Son3
{
public:
	void func()
	{
		//Son3是私有继承，所以继承Son3的属性在GrandSon3中都无法访问到
		//m_A;
		//m_B;
		//m_C;
	}
};
```

> 注意：父类中私有成员也是被子类继承下去了，只是由编译器给隐藏后访问不到
> 
1. 继承中的构造与析构

继承中 先调用父类构造函数，再调用子类构造函数，析构顺序与构造相反

#### **3.3.2 菱形继承**

**菱形继承概念：**

- 两个派生类继承同一个基类
- 又有某个类同时继承者两个派生类
- 这种继承被称为菱形继承，或者钻石继承

可能的问题：最下层子类使用数据时，可能产生二义性，而且继承来自最高基类的数据只需要一份。

为了解决上述问题，我们引入**虚继承**

```cpp
class Animal
{
public:
	int m_Age;
};

//继承前加virtual关键字后，变为虚继承
//此时公共的父类Animal称为虚基类
class Sheep : virtual public Animal {};
class Tuo   : virtual public Animal {};
class SheepTuo : public Sheep, public Tuo {};

void test01()
{
	SheepTuo st;
	st.Sheep::m_Age = 100;
	st.Tuo::m_Age = 200;

	cout << "st.Sheep::m_Age = " << st.Sheep::m_Age << endl;
	cout << "st.Tuo::m_Age = " <<  st.Tuo::m_Age << endl;
	cout << "st.m_Age = " << st.m_Age << endl;  //输出是最后被定义的父类数据st.Tuo::m_Age
}

```

### **3.4 多态**

多态满足条件： 
1、有继承关系
2、子类重写父类中的虚函数
多态使用：
父类指针或引用指向子类对象

#### **3.4.1 多态基本概念**

多态分为两类

- 静态多态: 函数重载 和 运算符重载属于静态多态，复用函数名
- 动态多态: 派生类和虚函数实现运行时多态

静态多态和动态多态区别：

- **静态多态的函数地址早绑定 - 编译阶段确定函数地址**
- **动态多态的函数地址晚绑定 - 运行阶段确定函数地址**

```cpp
class Animal
{
public:
	//Speak函数就是虚函数
	//函数前面加上virtual关键字，变成虚函数，那么编译器在编译的时候就不能确定函数调用了。
	virtual void speak()
	{
		cout << "动物在说话" << endl;
	}
};

class Cat :public Animal
{
public:
	void speak()
	{
		cout << "小猫在说话" << endl;
	}
};

class Dog :public Animal
{
public:

	void speak()
	{
		cout << "小狗在说话" << endl;
	}

};

//我们传入什么对象，那么就调用什么对象的函数
//如果函数地址在编译阶段就能确定，那么静态联编
//如果函数地址在运行阶段才能确定，就是动态联编
void DoSpeak(Animal & animal)
{
	animal.speak(); //传入什么对象，那么就调用什么对象的函数
}
void test01()
{
	Cat cat;
	DoSpeak(cat);
	Dog dog;
	DoSpeak(dog);
}
```

#### **3.4.2 纯虚函数**

在多态中，通常父类中虚函数的实现是毫无意义的，主要都是调用子类重写的内容

因此可以将虚函数改为**纯虚函数**

纯虚函数语法：`virtual 返回值类型 函数名 （参数列表）= 0 ;`

当类中有了纯虚函数，这个类也称为抽象类

```cpp
class Base
{
public:
	//纯虚函数
	//类中只要有一个纯虚函数就称为抽象类
	//抽象类无法实例化对象
	//子类必须重写父类中的纯虚函数，否则也属于抽象类
	virtual void func() = 0;
};
class Son :public Base
{
public:
	virtual void func()
	{
		cout << "func调用" << endl;
	};
};

void test01()
{
	Base * base = NULL;
	//base = new Base; // 错误，抽象类无法实例化对象
	base = new Son;
	base->func();
	delete base;//记得销毁
}
```

### **3.5 C++对象模型和this指针**

在C++中，类内的成员变量和成员函数分开存储

只有非静态成员变量才属于类的对象上，每一个非静态成员函数只会诞生一份函数实例，也就是说多个同类型的对象会共用一块代码

**this指针用于指向被调用的成员函数所属的对象**

#### **3.5.1 this指针作用**

- 当形参和成员变量同名时，可用this指针来区分
- 在类的非静态函数返回对象本身，可用`return *this`

```cpp
class Person
{
public:

	Person(int age)
	{
		//1、当形参和成员变量同名时，可用this指针来区分
		this->age = age;
	}

	Person& PersonAddPerson(Person p)
	{
		this->age += p.age;
		//返回对象本身
		return *this;
	}

	int age;
};
```

#### **3.5.2 const修饰成员函数**

**常函数：**

- 成员函数后加const后我们称为这个函数为**常函数**
- 常函数内不可以修改成员属性
- 成员属性声明时加关键字mutable后，在常函数中依然可以修改

**常对象：**

- 声明对象前加const称该对象为常对象
- 常对象只能调用常函数

```cpp
class Person {
public:
	Person() {
		m_A = 0;
		m_B = 0;
	}

	//this指针的本质是一个指针常量，指针的指向不可修改
	//如果想让指针指向的值也不可以修改，需要声明常函数
	void ShowPerson() const {
		//const Type* const pointer;
		//this = NULL; //不能修改指针的指向 Person* const this;
		//this->mA = 100; //但是this指针指向的对象的数据是可以修改的

		//const修饰成员函数，表示指针指向的内存空间的数据不能修改，除了mutable修饰的变量
		this->m_B = 100;
	}

	void MyFunc() const {
		//mA = 10000;  // 不可修改
	}

public:
	int m_A;
	mutable int m_B; //可修改 可变的
};

//const修饰对象  常对象
void test01() {

	const Person person; //常量对象
	cout << person.m_A << endl;
	//person.mA = 100; //常对象不能修改成员变量的值,但是可以访问
	person.m_B = 100; //但是常对象可以修改mutable修饰成员变量

	//常对象访问成员函数
	person.MyFunc(); //常对象不能调用const的函数

}
```

#### **3.5.3 静态成员**

静态成员就是在成员变量和成员函数前加上关键字`static`，称为静态成员

- 静态成员变量
    - **所有对象共享同一份数据**
    - 在编译阶段分配内存
    - **类内声明，类外初始化**
- 静态成员函数
    - 所有对象共享同一个函数
    - 静态成员函数只能访问静态成员变量

```cpp
class Person
{

public:

	static int m_A; //静态成员变量
    int m_C;

    static void func()
	{
		cout << "func调用" << endl;
		m_A = 100;
		//m_C = 100; //错误，不可以访问非静态成员变量
	}

private:
	static int m_B;  //类外不可访问private
};
int Person::m_A = 10;
int Person::m_B = 10;
```

### **3.6 友元**

友元的作用：就是让一个函数或者类 访问另一个类中私有成员

友元修饰符：`friend`

1. 全局函数做友元

```cpp
class Building
{
    //告诉编译器 goodGay全局函数 是 Building类的好朋友，可以访问类中的私有内容
    friend void goodGay(Building *building);

public:
    Building()
    {
        this->m_SittingRoom = "客厅";
        this->m_BedRoom = "卧室";
    }

public:
    string m_SittingRoom; //客厅

private:
    string m_BedRoom; //卧室
};

void goodGay(Building *building)
{
    cout << "好基友正在访问： " << building->m_SittingRoom << endl;
    cout << "好基友正在访问： " << building->m_BedRoom << endl;
}
//Building b;
//goodGay(&b)
```

1. 类做友元

```cpp
#include <iostream>
using namespace std;
class Building;
class goodGay
{
public:
    goodGay();
    void visit();

private:
    Building *building;
};

class Building
{
    //告诉编译器 goodGay类是Building类的好朋友，可以访问到Building类中私有内容
    friend class goodGay;

public:
    Building();

public:
    string m_SittingRoom; //客厅
private:
    string m_BedRoom; //卧室
};

Building::Building()
{
    this->m_SittingRoom = "客厅";
    this->m_BedRoom = "卧室";
}

goodGay::goodGay()
{
    building = new Building;
}

void goodGay::visit()
{
    cout << "好基友正在访问" << building->m_SittingRoom << endl;
    cout << "好基友正在访问" << building->m_BedRoom << endl;
}

void test01()
{
    goodGay gg;
    gg.visit();
}

int main()
{
    test01();
    return 0;
}
```

1. 成员函数做友元

```cpp
class Building;
class goodGay
{
public:
    goodGay();
    void visit();  //只让visit函数作为Building的好朋友，可以发访问Building中私有内容
    void visit2(); // visit2函数不可以访问Building中的私有内容

private:
    Building *building;
};

class Building
{
    //告诉编译器  goodGay类中的visit成员函数 是Building好朋友，可以访问私有内容
    friend void goodGay::visit();

public:
    Building();

public:
    string m_SittingRoom; //客厅
private:
    string m_BedRoom; //卧室
};

Building::Building()
{
    this->m_SittingRoom = "客厅";
    this->m_BedRoom = "卧室";
}

goodGay::goodGay()
{
    building = new Building;
}

void goodGay::visit()
{
    cout << "好基友正在访问" << building->m_SittingRoom << endl;
    cout << "好基友正在访问" << building->m_BedRoom << endl;
}

void goodGay::visit2()
{
    cout << "好基友正在访问" << building->m_SittingRoom << endl;
    // cout << "好基友正在访问" << building->m_BedRoom << endl;  //报错，不可访问
}

void test01()
{
    goodGay gg;
    gg.visit();
}
```

### **3.7 关于析构函数与构造函数的补充**

#### **3.7.1 C++中析构函数的作用**

- 析构函数是一个类的成员函数，名字由波浪号接类名`~student()`构成。它没有返回值，也不接受参数。由于析构函数不接受参数，因此它不能被重载。对于一个给定类，只会有唯一一个析构函数。
- 析构函数和构造函数对应，当对象结束其生命周期，**如对象所在的函数已调用完毕时，系统会自动执行析构函数**。析构函数释放对象使用的资源，并销毁对象的非static数据成员。
- 当一个类未定义自己的析构函数时，编译器会为它定义一个**合成析构函数**(即使自定义了析构函数，编译器也总是会为我们合成一个析构函数，并且如果自定义了析构函数，编译器在执行时会先调用自定义的析构函数再调用合成的析构函数）。

> 合成析构函数按对象创建时的逆序撤销每个非static成员
> 
- 对于某些类，合成析构函数被用来阻止该类型的对象被销毁。否则，合成析构函数的函数体就为空。因此，许多简单的类中没有用显式的析构函数。合成析构函数无法自动释放动态内存。 **如果一个类中有指针，且在使用的过程中动态的申请了内存，那么最好显式构造析构函数，在销毁类之前，释放掉申请的内存空间，避免内存泄漏。** 类析构顺序：1）派生类本身的析构函数；2）对象成员析构函数；3）基类析构函数。

#### **3.7.2 C++默认的析构函数为什么不是虚函数**

C++默认的析构函数不是虚函数是因为虚函数需要额外的虚函数表和虚表指针，占用额外的内存。对于不会被继承的类来说，其析构函数如果是虚函数，就会浪费内存。因此如果定义的类会被继承，一定要重新定义析构函数，并且设置为虚函数。

#### **3.7.3 存在派生类的基类析构函数为什么必须是虚函数**

- 对与一个基类和派生类来说，在调用构造函数时先基类的构造函数，再调用派生类的构造函数；而当调用析构函数时，则要先调用派生类再调用基类的析构函数。
- **如果定义了一个指向派生类对象的基类指针，当析构函数为普通函数时，释放该基类指针时，只会调用基类的析构函数，而不会调用派生类的析构函数，会导致内存泄漏。**
- **当基类析构函数被定义为虚函数时，在调用析构函数时，会在程序运行期间根据指向的对象类型到它的虚函数表中找到对应的虚函数(动态绑定)，此时找到的是派生类的析构函数，调用派生类析构函数之后再调用基类的析构函数，不会导致内存泄漏。**

#### **3.7.4 构造函数为什么不能是虚函数**

如果构造函数是虚函数，那么一定有一个已经存在的类对象obj，obj中的虚指针来指向虚表的构造函数地址（通过obj的虚指针来调用）；可是构造函数又是用来创建并初始化对象的，虚指针也是存储在对象的内存空间的。总的来说就是调用虚函数需要有类的对象，但是构造函数就是用来生成对象的，所以矛盾。

#### **3.7.5 静态函数和虚函数的区别**

静态函数在编译时就已确定运行时机，虚函数在运行的时候动态绑定。虚函数因为用了虚函数表机制，调用时会增加一次内存开销。

## **4. C++关键字： static const inline**

### **4.1 static关键字的作用**

1. **全局静态变量**
- 在全局变量前加上关键字`static`，全局变量就定义成一个全局静态变量
- 内存位置：静态存储区(全局区)，在整个程序运行期间一直存在

> 内存分区相关的知识见：C++学习笔记——1. 内存分区模型（代码区、全局区、栈区、堆区）
> 
- 初始化：未经初始化的全局静态变量会被自动初始化为0（自动对象的值是任意的，除非他被显式初始化），**编译时初始化**
- 作用域：全局静态变量在声明他的文件之外是不可见的，准确地说作用域是从定义之处开始，到文件结尾
1. **局部静态变量**
- 在局部变量之前加上关键字`static`，局部变量就成为一个局部静态变量
- 内存中的位置：静态存储区 (全局区)
- 初始化：静态局部变量在第一次使用时被首次初始化，即以后的函数调用不再进行初始化，未经初始化的会被程序自动初始化为0
- 作用域：作用域仍为局部作用域，当定义它的函数或者语句块结束的时候，作用域结束。
- 但是当局部静态变量离开作用域后，并没有销毁，而是仍然驻留在内存当中，只不过我们不能再对它进行访问，直到该函数再次被调用，并且值不变
1. **静态函数**
- 在函数返回类型前加`static`，函数就定义为静态函数
- 函数的定义和声明在默认情况下都是`extern`的，若函数使用static修饰，那么这个函数只可在本cpp内使用，不会同其他cpp中的同名函数引起冲突
- 注意：不要再头文件中声明static的全局函数,不要在cpp内声明非static的全局函数， 如果你要在多个cpp中复用该函数，就把它的声明提到头文件里去，否则cpp内部声明需加 上static修饰；
1. **类中的静态数据成员**
- 静态数据成员可以实现多个对象之间的数据共享，它是类的所有对象的共享成员，所有类共享同一份静态数据，如果改变它的值，则各对象中这个数据成员的值都被改变
- 静态数据成员是在创建类对象前被分配空间，到程序结束之后才释放，只要类中指定了静态数据成员，即使不定义对象，也会为静态数据成员分配空间
- 静态数据成员可以被初始化，但是只能在类体外进行初始化，静态成员变量使用前必须先初始化，若未对静态数据成员赋初值，则编译器会自动为其初始化为0
- 静态数据成员既可以通过对象名引用，也可以通过类名引用。 基类定义了static静态成员，则整个继承体系里只有一个这样的成员。无论派生出多少个子类，都有一个static成员实例
1. **类中的静态函数**
- 静态成员函数和静态数据成员一样，他们都属于类的静态成员，而不是对象成员
- 非静态成员函数有 this 指针，而静态成员函数没有this 指针
- 静态成员函数主要用来访问静态成员而不能访问非静态成员

### **4.2 const关键字的作用**

1. **const修饰普通变量**

使用const修饰普通变量，在定义该变量时，必须初始化，并且之后其值不会再改变。

1. **const的引用**

把引用绑定到const对象上，称之为 对常量引用,对常量引用不能被用作修改它所绑定的对象

```cpp
const int ci=1024;
const int &ri=ci; //正确
ri=42; //错误，r1是常量引用，不能修改其绑定对象
int &r2=ci; //错误，这样就可以通过r2改变ci，显然是错误的
```

> 引用的本质：常量指针
> 
> 
> int y=&x; 等价于 `int* const y = x;`
> 
1. **指针和const**
- 和引用一样，可以使用指针指向常量，这称为指向常量的指针，此时指针指向的是常量， 因此无法通过指针改变其指向对象的值，想要存放常量对象的地址，只能使用指向常量的指针`const int *ptr`。

```cpp
onst int a=1;
const int *ptr=&a; //正确，但是无法改变*ptr，因为ptr所指的是常量
int *p=&a; //错误，a为常量，所以只能使用指向常量的指针
```

- 除了指向常量的指针外，还可以使用从const修饰指针，即指针本身是常量，称为常量指针。常量指针必须初始化，并且之后该指针的值就不会再改变。

```cpp
int a=1;
int *const b=a; //此时b只能指向a，无法指向c等变量，但a不是常量，因此a还是
可以改变的，*b也可以改变，但是b无法改变，类似于常量引用。
```

1. **函数中的const参数**
- const修饰函数参数，表示参数不可变，此时可以使用const引用传递
- const引用传递和函数按值传递的效果是一样的，但按值传递会先建立一个类对象的副本, 然后传递过去,而它直接传递地址,所以这种传递比按值传递更高效

### **4.3 inline关键字的作用**

1. **inline的作用**

牺牲存储空间，减少内存空间使用

在c/c++中，为了解决一些**频繁调用的小函数大量消耗栈空间**(栈内存)的问题，引入了inline修饰符，表示为内联函数。

> 栈空间(栈区)：放置程序的局部数据（也就是函数内数据）的内存空间。由编译器自动分配释放, 存放函数的参数值,局部变量等 (函数作用后释放)
> 

增加了 inline 关键字的函数称为“内联函数”。内联函数和普通函数的区别在于：当编译器处理调用内联函数的语句时，不会将该语句编译成函数调用的指令，而是直接将整个函数体的代码插人调用语句处，就像整个函数体在调用处被重写了一遍一样。

有了内联函数，就能像调用一个函数那样方便地重复使用一段代码，而不需要付出执行函数调用的额外开销。很显然，**使用内联函数会使最终可执行程序的体积增加**。

1. **inline和宏定义的区别**
- 内联函数在编译时展开，宏在预编译时展开
- 内联函数直接嵌入到目标代码中，宏是简单的做文本替换
- 内联函数可以完成诸如类型检测，语句是否正确等编译功能，宏就不具有这样的功能
- 宏不是函数，inline函数是函数
- 宏在定义时要小心处理宏参数，一般用括号括起来，否则容易出现二义性。而内联函数不会出现二义性
1. **inline使用限制**

inline的使用是有所限制的，inline只适合涵数体内代码简单的涵数使用，不能包含复杂的结构控制语句例如while、switch，并且不能内联函数本身不能是直接递归函数（即，自己内部还调用自己的函数）

## **5. C++泛型编程——模板**

两种模板机制：**函数模板**和**类模板**

### **5.1 函数模板**

#### **5.1.1 语法**

```cpp
template<typename T>
函数声明或定义
```

**解释**：

`template`  ---  声明创建模板

`typename`  --- 表面其后面的符号是一种数据类型，可以用`class`代替

`T`    ---   通用的数据类型，名称可以替换，通常为大写字母

```cpp
template<typename T>
void mySwap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}
```

#### **5.1.2 函数模板使用**

使用函数模板有两种方式：自动类型推导、显示指定类型

```cpp
void test01()
{
	int a = 10;
	int b = 20;

	// 利用模板实现交换
	// 1、自动类型推导
	mySwap(a, b);

	// 2、显示指定类型
	mySwap<int>(a, b);

	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}
```

注意：

- **模板必须确定出T的数据类型才可以使用**

```cpp
void test01()
{
	int a = 10;
	int b = 20;
	char c = 'c';

	mySwap(a, b); // 正确，可以推导出一致的T
	//mySwap(a, c); // 错误，推导不出一致的T类型
}
```

- 自动类型推导，必须推导出一致的数据类型T,才可以使用

```cpp
template<class T>
void func()
{
	cout << "func 调用" << endl;
}

void test02()
{
	//func(); //错误，模板不能独立使用，必须确定出T的类型
	func<int>(); //利用显示指定类型的方式，给T一个类型，才可以使用该模板
}
```

### **5.2 类模板**

建立一个通用类，类中的成员 数据类型可以不具体制定，用一个**虚拟的类型**来代表。

#### **5.2.1 语法**

```cpp
template<typename T>
类
```

**解释**：

`template`  ---  声明创建模板

`typename`  --- 表面其后面的符号是一种数据类型，可以用`class`代替

`T`    ---   通用的数据类型，名称可以替换，通常为大写字母

#### **5.2.2 示例**

```cpp
template<class NameType, class AgeType>
class Person
{
public:
	Person(NameType name, AgeType age)
	{
		this->mName = name;
		this->mAge = age;
	}
	void showPerson()
	{
		cout << "name: " << this->mName << " age: " << this->mAge << endl;
	}
public:
	NameType mName;
	AgeType mAge;
};
```

#### **5.2.3 类模板与函数模板的区别**

- **类模板没有自动类型推导的使用方式，只能用显示指定类型方式**
- **类模板在模板参数列表中可以有默认参数**

```cpp
#include <string>
#include <iostream>
using namespace std;
// 类模板
template<class NameType, class AgeType = int>
class Person
{
public:
	Person(NameType name, AgeType age)
	{
		this->mName = name;
		this->mAge = age;
	}
	void showPerson()
	{
		cout << "name: " << this->mName << " age: " << this->mAge << endl;
	}
public:
	NameType mName;
	AgeType mAge;
};

// 1、类模板没有自动类型推导的使用方式
void test01()
{
	// Person p("孙悟空", 1000); // 错误 类模板使用时候，不可以用自动类型推导
	Person <string ,int>p("孙悟空", 1000); // 必须使用显示指定类型的方式，使用类模板
	p.showPerson();
}

// 2、类模板在模板参数列表中可以有默认参数
void test02()
{
	Person <string> p("猪八戒", 999); //类模板中的模板参数列表 可以指定默认参数
	p.showPerson();
}

int main()
{
	test01();
	test02();
	return 0;
}
```

#### **5.2.4 类模板对象做函数参数**

三种传入方式：

1. 指定传入的类型 --- 直接显示对象的数据类型
2. 参数模板化 --- 将对象中的参数变为模板进行传递
3. 整个类模板化 --- 将这个对象类型 模板化进行传递

```cpp
// 1、指定传入的类型
void printPerson1(Person<string, int> &p)
{
	p.showPerson();
}
void test01()
{
	Person <string, int >p("孙悟空", 100);
	printPerson1(p);
}

// 2、参数模板化
template <class T1, class T2>
void printPerson2(Person<T1, T2>&p)
{
	p.showPerson();
	cout << "T1的类型为： " << typeid(T1).name() << endl;
	cout << "T2的类型为： " << typeid(T2).name() << endl;
}
void test02()
{
	Person <string, int >p("猪八戒", 90);
	printPerson2(p);
}

// 3、整个类模板化
template<class T>
void printPerson3(T & p)
{
	cout << "T的类型为： " << typeid(T).name() << endl;
	p.showPerson();

}
void test03()
{
	Person <string, int >p("唐僧", 30);
	printPerson3(p);
}
```

> 输出：
> 
> 
> name: 孙悟空 age: 100
> name: 猪八戒 age: 90
> T1的类型为： NSt7**cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
> T2的类型为： i
> T的类型为： 6PersonINSt7**cxx1112basic_stringIcSt11char_traitsIcESaIcEEEiE
> name: 唐僧 age: 30
> 

#### **5.2.5 类模板与继承**

当类模板碰到继承时，需要注意一下几点：

- 当子类继承的父类是一个类模板时，**子类在声明的时候，要指定出父类中T的类型**
- 如果不指定，编译器无法给子类分配内存
- 如果想灵活指定出父类中T的类型，子类也需变为类模板

```cpp
template<class T>
class Base
{
	T m;
};

//class Son:public Base  //错误，c++编译需要给子类分配内存，必须知道父类中T的类型才可以向下继承
class Son :public Base<int> //必须指定一个类型
{
};
void test01()
{
	Son c;
}

//类模板继承类模板 ,可以用T2指定父类中的T类型
template<class T1, class T2>
class Son2 :public Base<T2>
{
public:
	Son2()
	{
		cout << typeid(T1).name() << endl;
		cout << typeid(T2).name() << endl;
	}
};

void test02()
{
	Son2<int, char> child1;
}

int main()
{
	test01();
	test02();
	return 0;
}
```

## **6. 文件操作**

程序运行时产生的数据都属于临时数据，程序一旦运行结束都会被释放

通过**文件可以将数据持久化**

C++中对文件操作需要包含头文件 `<fstream>`

文件类型分为两种：

1. **文本文件** - 文件以文本的**ASCII码**形式存储在计算机中
2. **二进制文件** - 文件以文本的**二进制**形式存储在计算机中

操作文件的三大类:

1. `ofstream`：写操作
2. `ifstream`： 读操作
3. `fstream`： 读写操作

### **6.1 文本文件**

#### **6.1.1 写文件**

写文件步骤如下：

1. 包含头文件 `#include <fstream>`
2. 创建流对象 `ofstream ofs;`
3. 打开文件 `ofs.open("文件路径",打开方式);`
4. 写数据 `ofs << "写入的数据";`
5. 关闭文件 `ofs.close();`

文件打开方式：

| 打开方式 | 解释 |
| --- | --- |
| ios::in | 为读文件而打开文件 |
| ios::out | 为写文件而打开文件 |
| ios::ate | 初始位置：文件尾 |
| ios::app | 追加方式写文件 |
| ios::trunc | 如果文件存在先删除，再创建 |
| ios::binary | 二进制方式 |

**注意：** 文件打开方式可以配合使用，利用|操作符

例如：用二进制方式写文件 `ios::binary |  ios:: out`

**示例：**

```cpp
#include <fstream> // 头文件

void test01()
{
	ofstream ofs; // 流对象
	ofs.open("test.txt", ios::out); // 打开文件

	ofs << "姓名：张三" << endl; // 写入数据
	ofs << "性别：男" << endl;
	ofs << "年龄：18" << endl;

	ofs.close(); // 关闭文件
}

int main()
{
	test01();
	return 0;
}
```

#### **6.1.2 读文件**

读文件步骤如下：

1. 包含头文件 `#include <fstream>`
2. 创建流对象 `ifstream ifs;`
3. 打开文件并判断文件是否打开成功 `ifs.open("文件路径",打开方式);`
4. 读数据 四种方式读取

> char buf[1024] = { 0 }; 
while (ifs >> buf); // 第一种
> 

> char buf[1024] = { 0 };
while(ifs.getline(buf,sizeof(buf))); // 第二种
> 

> string buf; 
while(getline(ifs, buf)); // 第三种
> 

> char c;
while((c = ifs.get()) != EOF); // 第四种
> 
1. 关闭文件 `ifs.close();`

```cpp
#include <fstream>
#include <string>
void test01()
{
	ifstream ifs;
	ifs.open("test.txt", ios::in);

	if (!ifs.is_open())
	{
		cout << "文件打开失败" << endl;
		return;
	}

	// 第一种
	//char buf[1024] = { 0 };
	//while (ifs >> buf)
	//{
	//	cout << buf << endl;
	//}

	// 第二种
	//char buf[1024] = { 0 };
	//while (ifs.getline(buf,sizeof(buf)))
	//{
	//	cout << buf << endl;
	//}

	// 第三种
	//string buf;
	//while (getline(ifs, buf))
	//{
	//	cout << buf << endl;
	//}

	char c;
	while ((c = ifs.get()) != EOF)
	{
		cout << c;
	}

	ifs.close();
}

int main()
{
	test01();
	return 0;
}
```

### **6.2 二进制文件**

以二进制的方式对文件进行读写操作

打开方式要指定为 `ios::binary`

#### **6.2.1 写文件**

二进制方式写文件主要利用流对象调用成员函数write

函数原型 ：`ostream& write(const char * buffer,int len);`

参数解释：字符指针buffer指向内存中一段存储空间, len 是读写的字节数

**示例：**

```cpp
#include <fstream> //1、包含头文件
#include <string>

class Person
{
public:
	char m_Name[64];
	int m_Age;
};

//二进制文件  写文件
void test01()
{
	//2、创建输出流对象
	ofstream ofs("person.txt", ios::out | ios::binary);

	//3、打开文件
	//ofs.open("person.txt", ios::out | ios::binary);

	Person p = {"张三"  , 18};

	//4、写文件
	ofs.write((const char *)&p, sizeof(p));

	//5、关闭文件
	ofs.close();
}

int main()
{
	test01();
	return 0;
}
```

> ofstream ofs("person.txt", ios::out | ios::binary);
> 
> 
> 等同于
> 
> ofstream ofs;
> 
> ofs.open("person.txt", ios::out | ios::binary);
> 

#### **6.2.2 读文件**

二进制方式读文件主要利用流对象调用成员函数read

函数原型：`istream& read(char *buffer,int len);`

参数解释：字符指针buffer指向内存中一段存储空间。len是读写的字节数

示例：

```cpp
#include <fstream>
#include <string>

class Person
{
public:
	char m_Name[64];
	int m_Age;
};

void test01()
{
	ifstream ifs("person.txt", ios::in | ios::binary);
	if (!ifs.is_open())
	{
		cout << "文件打开失败" << endl;
	}

	Person p;
	ifs.read((char *)&p, sizeof(p));

	cout << "姓名： " << p.m_Name << " 年龄： " << p.m_Age << endl;
}

int main()
{
	test01();
	return 0;
}
```

### **6.3 综合案例——罚抄100000遍对不起**

```cpp
#include <map>
#include <string>
#include <fstream>
#include <queue>

static const std::map<int, std::string> xuhao = {{1, "一"}, {2, "二"}, {3, "三"}, {4, "四"}, {5, "五"}, {6, "六"}, {7, "七"}, {8, "八"}, {9, "九"}, {10, "十"}, {100, "百"}, {1000, "千"}, {10000, "万"}};

std::string num2Chinese(int num)
{
    std::string cn;
    int n = num, pre = 0;
    for (auto it = xuhao.crbegin(); it != xuhao.crend(); it++) // 从后开始遍历iterator
    {
        int count = 0;
        for (; n >= it->first; n -= it->first)
            ++count;
        if (!count) // 如果还没找到最高位
            continue;
        if (pre / 10 > it->first)
            cn += "零";
        if (it->first >= 10 && (count != 1 || num > 20))
            cn += xuhao.at(count);
        cn += it->second;
        pre = it->first;
    }
    return cn;
}

int main()
{
    std::string all_line;
    for (int i = 1; i <= 100000; ++i)
        all_line += "对不起 第" + num2Chinese(i) + "遍\n";
    std::ofstream ofs("sorry.txt");
    ofs << all_line;
}
```

> 输出：
> 
> 
> 对不起 第一遍
> 对不起 第二遍
> 对不起 第三遍
> ...
> 对不起 第九万九千九百九十七遍
> 对不起 第九万九千九百九十八遍
> 对不起 第九万九千九百九十九遍
> 对不起 第十万遍
> 

## **7. STL(Standard Template Library)**

- C++的**面向对象**和**泛型编程**思想，目的就是**复用性的提升**
- 为了建立数据结构和算法的一套标准，诞生了**STL**(Standard Template Library,**标准模板库**)
- STL中几乎所有代码都采用了类模板和函数模板，使其适用于各种数据类型

### **7.0 STL基本概念**

STL大体分为六大组件，分别是:**容器、算法、迭代器、仿函数、适配器（配接器）、空间配置器**

1. 容器：各种数据结构，如vector、list、deque、set、map等,用来存放数据。
2. 算法：各种常用的算法，如sort、find、copy、for_each等
3. 迭代器：扮演了容器与算法之间的胶合剂。
4. 仿函数：行为类似函数，可作为算法的某种策略。
5. 适配器：一种用来修饰容器或者仿函数或迭代器接口的东西。
6. 空间配置器：负责空间的配置与管理。

#### **7.0.1 容器container**

**容器** 置物之所也

STL**容器**就是将运用**最广泛的一些数据结构**(例如：数组, 链表,树, 栈, 队列, 集合, 映射表 等）实现出来

这些容器分为**序列式容器**和**关联式容器**两种:

**序列式容器**：强调值的排序，序列式容器中的每个元素均有固定的位置。
	**关联式容器**：二叉树结构，各元素之间没有严格的物理上的顺序关系

#### **7.0.2 算法algorithm**

**算法** 问题之解法也

有限的步骤，解决逻辑或数学上的问题，这一门学科我们叫做算法(Algorithms)

算法分为:**质变算法**和**非质变算法**。

**质变算法**：是指运算过程中会更改区间内的元素的内容。例如拷贝，替换，删除等等

**非质变算法**：是指运算过程中不会更改区间内的元素内容，例如查找、计数、遍历、寻找极值等等

#### **7.0.3 迭代器iterator**

**迭代器** 容器和算法之间粘合剂

提供一种方法，使之能够依序寻访某个容器所含的各个元素，而又无需暴露该容器的内部表示方式。

每个容器都有自己专属的迭代器

迭代器种类：

| 种类 | 功能 | 支持运算 |
| --- | --- | --- |
| 输入迭代器 | 对数据的只读访问 | 只读，支持++、==、！= |
| 输出迭代器 | 对数据的只写访问 | 只写，支持++ |
| 前向迭代器 | 读写操作，并能向前推进迭代器 | 读写，支持++、==、！= |
| 双向迭代器 | 读写操作，并能向前和向后操作 | 读写，支持++、--， |
| 随机访问迭代器 | 读写操作，可以以跳跃的方式访问任意数据，功能最强的迭代器 | 读写，支持++、--、[n]、-n、<、<=、>、>= |

常用的容器中迭代器种类为双向迭代器，和随机访问迭代器

### **7.1 vector容器**

连续存储的容器，动态数组，在堆上分配空间。

底层实现：动态数组，内存分配是一段连续的空间。

成倍容量增长：vector 增加（插入）新元素时，如果未超过当时的容量，则还有剩余空间，那么直接添加到最后（插入指定位置），然后调整迭代器。如果没有剩余空间了，则会重新配置原有元素个数的两倍空间，然后将原空间元素通过复制的方式初始化新空间， 再向新空间增加元素，最后析构并释放原空间，之前的迭代器会失效。在VS下是1.5倍，在 GCC下是2倍。

在使用vector容器前先引入头文件`#include <vector>`

为了方便后续代码输出显示，这里定义一个输出函数`printVector`

```cpp
#include<iostream>
#include <vector>
using namespace std;
void printVector(vector<int>& v)
{
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++)
		cout << *it << " ";
	cout << endl;
}
```

#### **7.1.1 vector构造操作**

**函数原型：**

- `vector<T> v;` //采用模板实现类实现，默认构造函数
- `vector(v.begin(), v.end());` //将v[begin(), end())区间中的元素拷贝给本身。
- `vector(n, elem);` //构造函数将n个elem拷贝给本身。
- `vector(const vector &vec);` //拷贝构造函数。

```cpp
// 构造操作
void test01()
{
	vector<int> v1;
	for (int i = 0; i < 10; i++)
		v1.push_back(i);
	printVector(v1);

	vector<int> v2(v1.begin(), v1.end());
	printVector(v2);

	vector<int> v3(10, 100);
	printVector(v3);

	vector<int> v4(v3);
    printVector(v4);
}
```

#### **7.1.2 vector赋值操作**

**函数原型：**

- `vector& operator=(const vector &vec);`//重载等号操作符
- `assign(beg, end);` //将[beg, end)区间中的数据拷贝赋值给本身。
- `assign(n, elem);` //将n个elem拷贝赋值给本身。

```cpp
// 赋值操作
void test02()
{
	vector<int> v1;
	for (int i = 0; i < 10; i++)
		v1.push_back(i);
	printVector(v1);

	vector<int>v2;
	v2 = v1;
	printVector(v2);

	vector<int>v3;
	v3.assign(v1.begin(), v1.end());
    printVector(v3);

	vector<int>v4;
	v4.assign(10, 100);
    printVector(v4);
}
```

#### **7.1.3 vector容量和大小**

**函数原型：**

- `empty();` //判断容器是否为空
- `capacity();` //容器的容量
- `size();` //返回容器中元素的个数
- `resize(int num);` //重新指定容器的长度为num，若容器变长，则以默认值填充新位置。
    
    //如果容器变短，则末尾超出容器长度的元素被删除。
    
- `resize(int num, elem);` //重新指定容器的长度为num，若容器变长，则以elem值填充新位置。
    
    //如果容器变短，则末尾超出容器长度的元素被删除
    

```cpp
// 容量大小
void test03()
{
	vector<int> v1;
	for (int i = 0; i < 10; i++)
		v1.push_back(i);
    printVector(v1);

	if (v1.empty())
		cout << "v1为空" << endl;
	else
	{
		cout << "v1不为空" << endl;
		cout << "v1的容量 = " << v1.capacity() << endl;
		cout << "v1的大小 = " << v1.size() << endl;
	}

	//resize 重新指定大小 ，若指定的更大，默认用0填充新位置，可以利用重载版本替换默认填充
	v1.resize(15,10);
	printVector(v1);

	//resize 重新指定大小 ，若指定的更小，超出部分元素被删除
	v1.resize(5);
	printVector(v1);
}
```

#### **7.1.4 vector插入和删除**

**函数原型：**

- `push_back(ele);` //尾部插入元素ele
- `pop_back();` //删除最后一个元素
- `insert(const_iterator pos, ele);` //迭代器指向位置pos插入元素ele
- `insert(const_iterator pos, int count,ele);`//迭代器指向位置pos插入count个元素ele
- `erase(const_iterator pos);` //删除迭代器指向的元素
- `erase(const_iterator start, const_iterator end);`//删除迭代器从start到end之间的元素
- `clear();` //删除容器中所有元素

```cpp
// 插入和删除
void test04()
{
	vector<int> v1;
	// 尾插
	for (int i = 1; i < 6; i++)
		v1.push_back(i * 10);
	printVector(v1);
	// 尾删
	v1.pop_back();
	printVector(v1);
	// 插入
	v1.insert(v1.begin(), 100);
	printVector(v1);

	v1.insert(v1.begin(), 2, 1000);
	printVector(v1);

	//删除
	v1.erase(v1.begin());
	printVector(v1);

	//清空
	v1.erase(v1.begin(), v1.end());
	v1.clear();
	printVector(v1);
}
```

#### **7.1.5 vector数据存取**

**函数原型：**

- `at(int idx);` //返回索引idx所指的数据
- `operator[];` //返回索引idx所指的数据
- `front();` //返回容器中第一个数据元素
- `back();` //返回容器中最后一个数据元素

```cpp
void test05()
{
	vector<int>v1;
	for (int i = 0; i < 10; i++)
		v1.push_back(i);

	for (int i = 0; i < v1.size(); i++)
		cout << v1[i] << " ";
	cout << endl;

	for (int i = 0; i < v1.size(); i++)
		cout << v1.at(i) << " ";
	cout << endl;

	cout << "v1的第一个元素为： " << v1.front() << endl;
	cout << "v1的最后一个元素为： " << v1.back() << endl;
}
```

#### **7.1.6 vector互换容器**

**函数原型：**

- `swap(vec);` // 将vec与本身的元素互换

```cpp
void test06()
{
	vector<int>v1;
	for (int i = 0; i < 10; i++)
		v1.push_back(i);
	printVector(v1);

	vector<int>v2;
	for (int i = 10; i > 0; i--)
		v2.push_back(i);
	printVector(v2);

	// 互换容器
	cout << "互换后" << endl;
	v1.swap(v2);
	printVector(v1);
	printVector(v2);
}
```

#### **7.1.7 vector预留空间**

减少vector在**动态扩展容量**时的扩展次数

**函数原型：**

- `reserve(int len);`//容器预留len个元素长度，预留位置不初始化，元素不可访问。

```cpp
void test07()
{
    vector<int> v;
    v.reserve(50000); // 如果数据量较大，可以一开始利用reserve预留空间
    int num = 0;
    int *p = NULL;
    for (int i = 0; i < 100000; i++)
    {
        v.push_back(i);
        if (p != &v[0])
        {
            p = &v[0];
            num++;
        }
    }
    cout << "num:" << num << endl;
}
```

### **7.2 string容器**

string本质上是一个类

在使用string容器前先引入头文件`#include <string>`

**string和char * 区别：**

- char * 是一个指针
- string是一个类，类内部封装了char*，管理这个字符串，是一个char*型的容器。

#### **7.2.1 string构造函数**

**函数原型：**

- `string();`	//创建一个空的字符串 例如: string str;
`string(const char* s);` //使用字符串s初始化
- `string(const string& str);` //使用一个string对象初始化另一个string对象
- `string(int n, char c);` //使用n个字符c初始化

```cpp
// 构造函数
void test01()
{
    string s1; // 创建空字符串，调用无参构造函数
    cout << "str1 = " << s1 << endl;

    const char *str = "hello world";
    string s2(str); // 把c_string转换成了string

    cout << "str2 = " << s2 << endl;

    string s3(s2); //调用拷贝构造函数
    cout << "str3 = " << s3 << endl;

    string s4(10, 'a');
    cout << "str4 = " << s4 << endl;
}
```

#### **7.2.2 string赋值操作**

**函数原型：**

- `string& operator=(const char* s);` //char*类型字符串 赋值给当前的字符串
- `string& operator=(const string &s);` //把字符串s赋给当前的字符串
- `string& operator=(char c);` //字符赋值给当前的字符串
- `string& assign(const char *s);` //把字符串s赋给当前的字符串
- `string& assign(const char *s, int n);` //把字符串s的前n个字符赋给当前的字符串
- `string& assign(const string &s);` //把字符串s赋给当前字符串
- `string& assign(int n, char c);` //用n个字符c赋给当前字符串

```cpp
// 赋值
void test02()
{
    string str1;
    str1 = "hello world";
    cout << "str1 = " << str1 << endl;

    string str2;
    str2 = str1;
    cout << "str2 = " << str2 << endl;

    string str3;
    str3 = 'a';
    cout << "str3 = " << str3 << endl;

    string str4;
    str4.assign("hello c++");
    cout << "str4 = " << str4 << endl;

    string str5;
    str5.assign("hello c++", 5);
    cout << "str5 = " << str5 << endl;

    string str6;
    str6.assign(str5);
    cout << "str6 = " << str6 << endl;

    string str7;
    str7.assign(5, 'x');
    cout << "str7 = " << str7 << endl;
}
```

#### **7.2.3 string字符串拼接**

**函数原型：**

- `string& operator+=(const char* str);` //重载+=操作符
- `string& operator+=(const char c);` //重载+=操作符
- `string& operator+=(const string& str);` //重载+=操作符
- `string& append(const char *s);` //把字符串s连接到当前字符串结尾
- `string& append(const char *s, int n);` //把字符串s的前n个字符连接到当前字符串结尾
- `string& append(const string &s);` //同operator+=(const string& str)
- `string& append(const string &s, int pos, int n);`//字符串s中从pos开始的n个字符连接到字符串结尾

```cpp
// 字符串拼接
void test03()
{
    string str1;
    str1 = "hello world";
    cout << "str1 = " << str1 << endl;

    string str2;
    str2 = str1;
    cout << "str2 = " << str2 << endl;

    string str3;
    str3 = 'a';
    cout << "str3 = " << str3 << endl;

    string str4;
    str4.assign("hello c++");
    cout << "str4 = " << str4 << endl;

    string str5;
    str5.assign("hello c++", 5);
    cout << "str5 = " << str5 << endl;

    string str6;
    str6.assign(str5);
    cout << "str6 = " << str6 << endl;

    string str7;
    str7.assign(5, 'x');
    cout << "str7 = " << str7 << endl;
}
```

#### **7.2.4 string查找和替换**

**函数原型：**

- `int find(const string& str, int pos = 0) const;` //查找str第一次出现位置,从pos开始查找
- `int find(const char* s, int pos = 0) const;` //查找s第一次出现位置,从pos开始查找
- `int find(const char* s, int pos, int n) const;` //从pos位置查找s的前n个字符第一次位置
- `int find(const char c, int pos = 0) const;` //查找字符c第一次出现位置
- `int rfind(const string& str, int pos = npos) const;` //查找str最后一次位置,从pos开始查找
- `int rfind(const char* s, int pos = npos) const;` //查找s最后一次出现位置,从pos开始查找
- `int rfind(const char* s, int pos, int n) const;` //从pos查找s的前n个字符最后一次位置
- `int rfind(const char c, int pos = 0) const;` //查找字符c最后一次出现位置
- `string& replace(int pos, int n, const string& str);` //替换从pos开始n个字符为字符串str
- `string& replace(int pos, int n,const char* s);` //替换从pos开始的n个字符为字符串s

```cpp
// 查找和替换
void test041()
{
    // 查找
    // find查找是从左往后，rfind从右往左
    string str1 = "abcdefgde";

    int pos = str1.find("de"); // 查找str第一次出现位置,从pos开始查找

    if (pos == -1)
    {
        cout << "未找到" << endl;
    }
    else
    {
        cout << "pos = " << pos << endl;
    }

    pos = str1.rfind("de"); // 从pos查找s的前n个字符最后一次位置

    cout << "pos = " << pos << endl;
}

void test042()
{
    //替换
    string str1 = "abcdefgde";
    str1.replace(1, 3, "1111"); // 替换从pos开始n个字符为字符串str

    cout << "str1 = " << str1 << endl;
}
```

#### **7.2.5 string字符串比较**

**比较方式：**

字符串比较是按字符的ASCII码进行对比

| 比较结果 | 返回 |
| --- | --- |
| = | 0 |
| > | 1 |
| < | -1 |

**函数原型：**

- `int compare(const string &s) const;` //与字符串s比较
- `int compare(const char *s) const;` //与字符串s比较

```cpp
// 字符串比较
void test05()
{

    string s1 = "hello";
    string s2 = "aello";

    int ret = s1.compare(s2);

    if (ret == 0)
    {
        cout << "s1 等于 s2" << endl;
    }
    else if (ret > 0)
    {
        cout << "s1 大于 s2" << endl;
    }
    else
    {
        cout << "s1 小于 s2" << endl;
    }
}
```

#### **7.2.6 string字符存取**

**函数原型：**

- `char& operator[](int n);` //通过[]方式取字符
- `char& at(int n);` //通过at方法获取字符

```cpp
void test06()
{
    string str = "hello world";

    for (int i = 0; i < str.size(); i++)
    {
        cout << str[i] << " ";
    }
    cout << endl;

    // str.at(pos)
    for (int i = 0; i < str.size(); i++)
    {
        cout << str.at(i) << " ";
    }
    cout << endl;

    //字符修改
    str[0] = 'x';
    str.at(1) = 'x';
    cout << str << endl;
}
```

#### **7.2.7 string插入和删除**

**函数原型：**

- `string& insert(int pos, const char* s);` //插入字符串
- `string& insert(int pos, const string& str);` //插入字符串
- `string& insert(int pos, int n, char c);` //在指定位置插入n个字符c
- `string& erase(int pos, int n = npos);` //删除从Pos开始的n个字符

```cpp
void test07()
{
    string str = "hello";
    str.insert(1, "222"); // 插入
    cout << str << endl;

    str.erase(1, 3); // 删除:从1号位置开始3个字符
    cout << str << endl;
}
```

#### **7.2.8 string子串**

**函数原型：**

- `string substr(int pos = 0, int n = npos) const;` //返回由pos开始的n个字符组成的字符串

```cpp
void test08()
{

    string str = "abcdefg";
    string subStr = str.substr(1, 3);
    cout << "subStr = " << subStr << endl;

    string email = "hello@sina.com";
    int pos = email.find("@"); // 找不到则返回-1
    string username = email.substr(0, pos);
    cout << "username: " << username << endl;
    int pre = email.find("o");
    int las = email.rfind("o");
    cout << pre << " " << las << endl;
}
```

### **7.3 deque容器**

**功能：**

- 双端数组，可以对头端进行插入删除操作

**deque容器的迭代器支持随机访问的**

**deque与vector区别：**

- vector对于头部的插入删除效率低，数据量越大，效率越低
- deque相对而言，对头部的插入删除速度会比vector快
- vector访问元素时的速度会比deque快,这和两者内部实现有关

在使用deque容器前先引入头文件`#include <deque>`

为了方便显示，先定义一个输出函数`printDeque`

```cpp
#include <deque>

void printDeque(const deque<int>& d)
{
	for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++)
		cout << *it << " ";
	cout << endl;
}
```

#### **7.3.1 deque构造函数**

**函数原型：**

- `deque<T>` deqT; //默认构造形式
- `deque(beg, end);` //构造函数将[beg, end)区间中的元素拷贝给本身。
- `deque(n, elem);` //构造函数将n个elem拷贝给本身。
- `deque(const deque &deq);` //拷贝构造函数

```cpp
void test01()
{
    deque<int> d1;
    for (int i = 0; i < 10; i++)
    {
        d1.push_back(i);
    }
    printDeque(d1);

    deque<int> d2(d1.begin(), d1.end());
    printDeque(d2);

    deque<int> d3(10, 100);
    printDeque(d3);

    deque<int> d4 = d3;
    printDeque(d4);
}
```

#### **7.3.2 deque赋值操作**

**函数原型：**

- `deque& operator=(const deque &deq);` //重载等号操作符
- `assign(beg, end);` //将[beg, end)区间中的数据拷贝赋值给本身。
- `assign(n, elem);` //将n个elem拷贝赋值给本身。

```cpp
void test02()
{
    deque<int> d1;
    for (int i = 0; i < 10; i++)
    {
        d1.push_back(i);
    }
    printDeque(d1);

    deque<int> d2;
    d2 = d1;
    printDeque(d2);

    deque<int> d3;
    d3.assign(d1.begin(), d1.end());
    printDeque(d3);

    deque<int> d4;
    d4.assign(10, 100);
    printDeque(d4);
}
```

#### **7.3.3 deque大小操作**

**函数原型：**

- `deque.empty();` //判断容器是否为空
- `deque.size();` //返回容器中元素的个数
- `deque.resize(num);` //重新指定容器的长度为num,若容器变长，则以默认值填充新位置。
    
    //如果容器变短，则末尾超出容器长度的元素被删除。
    
- `deque.resize(num, elem);` //重新指定容器的长度为num,若容器变长，则以elem值填充新位置。
    
    //如果容器变短，则末尾超出容器长度的元素被删除。
    

```cpp
void test03()
{
    deque<int> d1;
    for (int i = 0; i < 10; i++)
    {
        if (i >= 5)
        {
            d1.push_back(i);
        }
        else
            d1.push_front(i);
    }
    printDeque(d1);

    //判断容器是否为空
    if (d1.empty())
    {
        cout << "d1为空!" << endl;
    }
    else
    {
        cout << "d1不为空!" << endl;
        //统计大小
        cout << "d1的大小为：" << d1.size() << endl;
    }

    //重新指定大小
    d1.resize(15, 1);
    printDeque(d1);

    d1.resize(5);
    printDeque(d1);
}
```

#### **7.3.4 deque插入和删除**

**函数原型：**

两端插入操作：

- `push_back(elem);` //在容器尾部添加一个数据
- `push_front(elem);` //在容器头部插入一个数据
- `pop_back();` //删除容器最后一个数据
- `pop_front();` //删除容器第一个数据

指定位置操作：

- `insert(pos,elem);` //在pos位置插入一个elem元素的拷贝，返回新数据的位置。
- `insert(pos,n,elem);` //在pos位置插入n个elem数据，无返回值。
- `insert(pos,beg,end);` //在pos位置插入[beg,end)区间的数据，无返回值。
- `clear();` //清空容器的所有数据
- `erase(beg,end);` //删除[beg,end)区间的数据，返回下一个数据的位置。
- `erase(pos);` //删除pos位置的数据，返回下一个数据的位置。

```cpp
//两端操作
void test041()
{
    deque<int> d;
    //尾插
    d.push_back(10);
    d.push_back(20);
    //头插
    d.push_front(100);
    d.push_front(200);

    printDeque(d);

    //尾删
    d.pop_back();
    //头删
    d.pop_front();
    printDeque(d);
}

//插入
void test042()
{
    deque<int> d;
    d.push_back(10);
    d.push_back(20);
    d.push_front(100);
    d.push_front(200);
    printDeque(d);

    d.insert(d.begin(), 1000);
    printDeque(d);

    d.insert(d.begin(), 2, 10000);
    printDeque(d);

    deque<int> d2;
    d2.push_back(1);
    d2.push_back(2);
    d2.push_back(3);

    d.insert(d.begin(), d2.begin(), d2.end());
    printDeque(d);
}

//删除
void test043()
{
    deque<int> d;
    d.push_back(10);
    d.push_back(20);
    d.push_front(100);
    d.push_front(200);
    printDeque(d);

    d.erase(d.begin());
    printDeque(d);

    d.erase(d.begin(), d.end());
    d.clear();
    printDeque(d);
}
```

#### **7.3.5 deque数据存取**

**函数原型：**

- `at(int idx);` //返回索引idx所指的数据
- `operator[];` //返回索引idx所指的数据
- `front();` //返回容器中第一个数据元素
- `back();` //返回容器中最后一个数据元素

```cpp
//数据存取
void test05()
{

    deque<int> d;
    d.push_back(10);
    d.push_back(20);
    d.push_front(100);
    d.push_front(200);

    for (int i = 0; i < d.size(); i++)
    {
        cout << d[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < d.size(); i++)
    {
        cout << d.at(i) << " ";
    }
    cout << endl;

    cout << "front:" << d.front() << endl;

    cout << "back:" << d.back() << endl;
}
```

#### **7.3.6 deque排序**

**算法：**

- `sort(iterator beg, iterator end)` //对beg和end区间内元素进行排序

```cpp
#include <iostream>
#include <deque>
#include <algorithm>
using namespace std;

void printDeque(const deque<int> &d)
{
    for (deque<int>::const_iterator it = d.begin(); it != d.end(); it++)
    {
        cout << *it << " ";
    }
    cout << endl;
}

void test07()
{

    deque<int> d;
    d.push_back(10);
    d.push_back(20);
    d.push_front(100);
    d.push_front(200);

    printDeque(d);
    sort(d.begin(), d.end());
    printDeque(d);
}

int main()
{
    test07();
    return 0;
}
```

### **7.4 stack 和 queue容器**

#### **7.4.1 stack容器**

**概念：**stack是一种**先进后出**(First In Last Out,FILO)的数据结构，只有一个出口

栈中只有顶端的元素才可以被外界使用，因此栈不允许有遍历行为

栈中进入数据称为  --- **入栈**  `push`

栈中弹出数据称为  --- **出栈**  `pop`

| 接口 | 函数 |
| --- | --- |
| 构造函数 | stack<T> stk;                                 //stack采用模板类实现， stack对象的默认构造形式                                                                                      stack(const stack &stk);            //拷贝构造函数 |
| 赋值操作 | stack& operator=(const stack &stk);           //重载等号操作符 |
| 数据存取 | push(elem);      //向栈顶添加元素                                                                                                                                                                pop();                //从栈顶移除第一个元素                                                                                                                                                                                                                      top();                //返回栈顶元素 |
| 大小操作 | empty();            //判断堆栈是否为空                                                                                                                                                                                 size();              //返回栈的大小 |

示例：

```cpp
#include <iostream>
#include <stack>
#include <algorithm>
using namespace std;

// FILO:先进后出

//栈容器常用接口
void test01()
{
    //创建栈容器 栈容器必须符合先进后出
    stack<int> s;

    //向栈中添加元素，叫做 压栈 入栈
    s.push(10);
    s.push(20);
    s.push(30);
    cout << "此时栈中元素的个数为:" << s.size() << endl;

    while (!s.empty())
    {
        //输出栈顶元素
        cout << "栈顶元素为： " << s.top() << endl;
        //弹出栈顶元素
        s.pop();
    }
    cout << "栈的大小为：" << s.size() << endl;
}

int main()
{
    test01();
    return 0;
}
```

#### **7.4.2 queue容器**

**概念：**Queue是一种**先进先出**(First In First Out,FIFO)的数据结构，它有两个出口

队列容器允许从一端新增元素，从另一端移除元素

队列中只有队头和队尾才可以被外界使用，因此队列不允许有遍历行为

队列中进数据称为 --- **入队**    `push`

队列中出数据称为 --- **出队**    `pop`

| 接口 | 函数 |
| --- | --- |
| 构造函数 | queue<T> que;                                 //queue采用模板类实现，queue对象的默认构造形式                                                                                     queue(const queue &que);            //拷贝构造函数 |
| 赋值操作 | queue& operator=(const queue &que);           //重载等号操作符 |
| 数据存取 | push(elem);      //向栈顶添加元素                                                                                                                                                                                                pop();                //从栈顶移除第一个元素                                                                                                                                                                                                                        back();                                    //返回最后一个元素                                                                                                                                                                             front();                                  //返回第一个元素 |
| 大小操作 | empty();            //判断堆栈是否为空                                                                                                                                                                                 size();              //返回栈的大小 |

示例：

```cpp
#include <queue>
#include <string>
#include <iostream>
using namespace std;
class Person
{
public:
    Person(string name, int age)
    {
        this->m_Name = name;
        this->m_Age = age;
    }

    string m_Name;
    int m_Age;
};

void test01()
{

    //创建队列
    queue<Person> q;

    //准备数据
    Person p1("唐僧", 30);
    Person p2("孙悟空", 1000);
    Person p3("猪八戒", 900);
    Person p4("沙僧", 800);

    //向队列中添加元素  入队操作
    q.push(p1);
    q.push(p2);
    q.push(p3);
    q.push(p4);

    //队列不提供迭代器，更不支持随机访问
    while (!q.empty())
    {
        //输出队头元素
        cout << "队头元素-- 姓名： " << q.front().m_Name
             << " 年龄： " << q.front().m_Age << endl;

        cout << "队尾元素-- 姓名： " << q.back().m_Name
             << " 年龄： " << q.back().m_Age << endl;

        cout << endl;
        //弹出队头元素
        q.pop();
    }

    cout << "队列大小为：" << q.size() << endl;
}

int main()
{
    test01();
    return 0;
}
```
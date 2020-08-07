package com.company.juc.jucdesign.applicationcontext;

import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class ThreadLocalExample {
    /** ThreadLocal 常用方法 initialValue 初始化参数 set get
     *  ThreadLocal 是用于解决线程内参数分配
     *  ThreadLocal.withInitial(Object::new);
    */
    public static void main(String[] args) {
        ThreadLocal<Integer> threadLocal=new ThreadLocal();
        IntStream.range(0,10).forEach(i -> new Thread(() -> {
            try{
                /** ThreadLocal
                 *  set 方法 流程
                 * 1. 获取当前线程 Thread.currentThread
                 * 2. 获取与之关联的ThreadLocalMap
                 * 3. 若map为空->4,否->5
                 * 4. map为null,开始创建，用当前实例作为key，
                 * 要存取的数据作为Value，对应到ThreadLocalMap中则是创建了一个Entry
                 * 5. set方法遍历map的整个Entry,若发现ThreadLocal相同,则使用新的数据替换，返回
                 * 6. 遍历过程中若Entry的key为null，则是将其逐出 防止内存泄漏
                 * 7. 创建新的entry,ThreadLocal作为Key，数据作为Value
                 * 8. 进行清理
                */
                threadLocal.set(i);
                /** ThreadLocal get方法
                 * 大致和set方法流程相同 但是若为空需要进行初始化 
                */
                System.out.println(Thread.currentThread()+"set-"+threadLocal.get());
                TimeUnit.SECONDS.sleep(1);
                System.out.println(Thread.currentThread()+"get-"+threadLocal.get());
                
            } catch (InterruptedException e) {
                //TODO: handle exception
                e.printStackTrace();
            }
        }).start());
    }
}
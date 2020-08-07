package com.company.juc.waitnotify;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class AwaitSignalSample {
    private Lock lock=new ReentrantLock();
    private Condition condition= lock.newCondition();

    private void before(){
        lock.lock();
        try{
            System.out.println("before");
            condition.signalAll();
        }finally {
            lock.unlock();
        }
    }
    private void after(){
        //tryLock 只会尝试性的去解锁
        lock.lock();
        try {
            condition.await();
            System.out.println("after");
        } catch (InterruptedException e) {
            e.printStackTrace();
        }finally {
            lock.unlock();
        }
    }

    public void run(){
        System.out.println("==========");
        ExecutorService executorService= Executors.newCachedThreadPool();
        executorService.submit(()->after());
        executorService.submit(()->before());
        executorService.shutdown();
    }
}

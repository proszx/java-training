package com.company.juc.lock;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class LockSample {
    //锁优化
    //自旋锁
    //锁消除
    private Lock lock=new ReentrantLock();
    public  void Lock(){
        lock.tryLock();
        try {
            for (int i=0;i<10;i++) {
                System.out.println(i + " ");
            }
        }finally {
            lock.unlock();
        }
    }
}

package com.company.juc.waitnotify;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class WaitNotifyExample {
    private synchronized  void before(){
        System.out.println("before");
        notifyAll();
    }
    private synchronized  void after(){
        try {
            wait();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("after");
    }
    public void run() throws InterruptedException {
        ExecutorService executorService= Executors.newCachedThreadPool();
        executorService.submit(()->before());
        executorService.submit(()->after());

        System.out.println("wait noti test");
        Thread.sleep(100);
        executorService.submit(()->after());
        executorService.submit(()->before());

        executorService.shutdown();


    }
}

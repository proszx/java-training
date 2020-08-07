package com.company.juc.aqs;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

public class SemaphoreExample {
    public  void run(){
        Semaphore semaphore=new Semaphore(4);
        ExecutorService executorService= Executors.newCachedThreadPool();
        for(int i=0;i<10;i++){
            executorService.submit(()->{
                try {
                    semaphore.acquire();
                    System.out.println(semaphore.availablePermits()+"===");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }finally {
                    semaphore.release();
                }

            });
        }
        executorService.shutdown();
    }
}

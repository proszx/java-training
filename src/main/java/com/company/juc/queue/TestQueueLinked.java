package com.company.juc.queue;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class TestQueueLinked {
    public static void main(String[] args) throws InterruptedException {
        int people=1000;
        int table=10;
        ConcurrentLinkedQueue<String> queue=new ConcurrentLinkedQueue<>();
        CountDownLatch countDownLatch=new CountDownLatch(table);
        for(int i=0;i<people;i++){
            queue.offer("people:"+(i+1));
        }
        ExecutorService executorService= Executors.newFixedThreadPool(table);
        for(int i=0;i<table;i++){
            executorService.submit(new Runnable() {
                @Override
                public void run() {
                    while(!queue.isEmpty()){
                        queue.poll();
                    }
                    countDownLatch.countDown();
                }
            });
        }
        countDownLatch.await();
        executorService.shutdownNow();
    }
}

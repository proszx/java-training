package com.company.juc.aqs;

import java.util.concurrent.*;

public class CountdownLatchSample {
    public  void run() throws InterruptedException {
        int cnt=10;
        CountDownLatch countDownLatch=new CountDownLatch(cnt);
        ExecutorService executorService= Executors.newCachedThreadPool();
        for(int i=0;i<cnt;i++){
            executorService.execute(()->{
                System.out.print("running ..");
                countDownLatch.countDown();
            });
        }
        countDownLatch.await();
        System.out.println("end");
        executorService.shutdown();
    }
    public void CylicBarrierrun(){
        int cnt=10;
        CyclicBarrier cyclicBarrier=new CyclicBarrier(10);
        ExecutorService executorService=Executors.newCachedThreadPool();
        for(int i=0;i<10;i++){
            executorService.execute(()->{
                System.out.print("before..");
                try {
                    cyclicBarrier.await();
                } catch (InterruptedException|BrokenBarrierException e){
                    e.printStackTrace();
                }
                System.out.println("after..");
            });
        }
        executorService.shutdown();
    }
}

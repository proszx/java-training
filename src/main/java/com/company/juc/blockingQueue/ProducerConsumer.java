package com.company.juc.blockingQueue;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.function.Consumer;

public class ProducerConsumer {
    private  static BlockingQueue<String> queue=new ArrayBlockingQueue<>(5);

    private static  class Producer extends  Thread{
        @Override
        public void run() {
            try {
                queue.put("product");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.print("producing");
        }
    }
    private static  class Consumer extends  Thread{
        @Override
        public void run() {
            try {
                queue.take();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.print("consuming");
        }
    }
    public void run(){
        for(int i=0;i<5;i++){
            Producer producer=new Producer();
            producer.start();
        }
        for(int i=0;i<2;i++){
            Consumer consumer=new Consumer();
            consumer.start();
        }
        for(int i=0;i<6;i++){
            Producer producer=new Producer();
            producer.start();
        }
    }
}

package com.company.juc.queue;

import java.util.concurrent.*;

public class TestDelayQueue {
    static  class Item implements Delayed{
        private long time;
        String name;
        public  Item(String name,long time,TimeUnit unit){
            this.name=name;
            this.time=System.currentTimeMillis()+(time>0?unit.toMillis(time):0);
        }
        @Override
        public long getDelay(TimeUnit unit) {
            return time-System.currentTimeMillis();
        }

        @Override
        public int compareTo(Delayed o) {
            Item item=(Item)o;
            long diff=this.time-item.time;
            if(diff<=0){
                return -1;
            }else{
                return 1;
            }

        }
    }
    public static void main(String[] args) throws InterruptedException {
        int people=1000;
        int table=10;
        ConcurrentLinkedDeque<String> deque=new ConcurrentLinkedDeque<>();
        CountDownLatch countDownLatch=new CountDownLatch(10);
        for(int i=0;i<people;i++){
            deque.offer("people"+(i+1));
        }
        ExecutorService executorService= Executors.newFixedThreadPool(table);
        for(int i=0;i<table;i++){
            executorService.submit(new Runnable() {
                @Override
                public synchronized void run() {
                    while(!deque.isEmpty()){
                        System.out.println(deque.poll());
                    }
                    countDownLatch.countDown();
                }
            });
        }
        countDownLatch.await();
        executorService.shutdown();
    }
}

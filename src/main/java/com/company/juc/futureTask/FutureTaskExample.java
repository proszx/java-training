package com.company.juc.futureTask;

import java.util.concurrent.*;

public class FutureTaskExample {
    public FutureTask run(){
        FutureTask<Integer> ft=new FutureTask<Integer>(new Callable<Integer>() {
            @Override
            public Integer call() {
                int result=0;
                for(int i=0;i<10;i++){
                    try {
                        Thread.sleep(1000);
                        result+=i;
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                return result;
            }
        });
        return ft;
    }


}

package com.company.juc.thread;

public class InterruptedSample  extends Thread{
    @Override
    public void run() {
        try {
            while(!interrupted()){
                Thread.sleep(2000);
                System.out.println("i was not interrupted");
            }
            Thread.sleep(2000);
            System.out.println("i was interrupted");
        }catch (InterruptedException e){
            e.printStackTrace();
        }
    }

}

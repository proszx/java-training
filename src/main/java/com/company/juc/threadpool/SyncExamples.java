package com.company.juc.threadpool;

public class SyncExamples {
    public void lockBlock(){
        synchronized (this){
            for (int i=0;i<10;i++) {
                System.out.print(i+" ");
            }
        }
    }
    public synchronized void lockFunc(){
        for (int i=0;i<10;i++) {
            System.out.print(i+" ");
        }
    }
    public synchronized static void lockStatic(){
        for (int i=0;i<10;i++) {
            System.out.print(i+" ");
        }
    }
    public void lockClazz(){
        synchronized (SyncExamples.class){
            for (int i=0;i<10;i++) {
                System.out.print(i+" ");
            }
        }
    }
}

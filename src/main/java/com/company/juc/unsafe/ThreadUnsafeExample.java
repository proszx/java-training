package com.company.juc.unsafe;

import java.util.concurrent.atomic.AtomicInteger;

public class ThreadUnsafeExample {
    private int cnt=0;
    private  volatile  int count=0;
    private int syncCount=0;
    private AtomicInteger atoCnt=new AtomicInteger(0);
    public void setCnt() {
        cnt++;
    }

    public  int getCnt(){
        return cnt;
    }

    public void setCount() {
        count++;
    }
    public  int getCount(){
        return  count;
    }
    public synchronized void setSyncCount() {
        syncCount++;
    }
    public  int getSyncCount(){
        return  syncCount;
    }
    public  void setAtoCnt(){
        atoCnt.incrementAndGet();
    }
    public int getAtoCnt(){
        return atoCnt.get();
    }
}

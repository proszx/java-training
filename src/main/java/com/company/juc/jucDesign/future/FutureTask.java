package com.company.juc.jucdesign.future;

public class FutureTask<T> implements Future<T>{
    private T result;
    private  boolean isDone=false;
    private final Object Lock=new Object();

    @Override
    public T get() throws InterruptedException {
        // TODO Auto-generated method stub
    //    return null;
        synchronized(Lock){
            while(!isDone){
                Lock.wait();
            }
            return result;
        }
    }
    protected void finish(T result){
        synchronized(Lock){
            if(isDone)
                return;
            this.result=result;
            this.isDone=true;
            Lock.notifyAll();
        }
    }
    @Override
    public boolean done() {
        // TODO Auto-generated method stub
        return isDone;
    }

}
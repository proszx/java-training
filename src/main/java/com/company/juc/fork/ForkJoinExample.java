package com.company.juc.fork;

import java.util.concurrent.RecursiveTask;

public class ForkJoinExample extends RecursiveTask<Integer> {
    private final  int threshold=5;
    private int first;
    private int last;
    public ForkJoinExample(int first,int last){
        this.first=first;
        this.last=last;
    }

    @Override
    protected Integer compute() {
        int result=0;
        if(last-first<threshold){
            for(int i=first;i<=last;i++){
                result+=i;
            }
        }else{
            int mid=first+(last-first)/2;
            ForkJoinExample leftTask=new ForkJoinExample(first,mid);
            ForkJoinExample rightTask=new ForkJoinExample(mid+1,last);
            leftTask.fork();
            rightTask.fork();
            result=leftTask.join()+rightTask.join();
        }
        return result;
    }
}

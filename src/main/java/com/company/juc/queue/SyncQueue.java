package com.company.juc.queue;

import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class SyncQueue {
    private volatile  static AtomicInteger atomicInteger=new AtomicInteger(10);
    public static void main(String[] args) {

        SynchronousQueue<String> synchronousQueue=new SynchronousQueue<>();

    }
}

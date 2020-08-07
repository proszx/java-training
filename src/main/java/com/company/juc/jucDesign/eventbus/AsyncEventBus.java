//package com.company.juc.jucdesign.eventbus;
//
//import java.util.concurrent.ThreadPoolExecutor;
//
//public class AsyncEventBus extends EventBus{
//    AsyncEventBus(String busName,EventExceptionHandler exceptionHandler,ThreadPoolExecutor executor){
//        super(busName,exceptionHandler,executor);
//    }
//    public AsyncEventBus(String busName,ThreadPoolExecutor executor){
//        this(busName,null,executor);
//    }
//    public AsyncEventBus(ThreadPoolExecutor executor){
//        this("default-async",null,executor);
//    }
//    public AsyncEventBus(EventExceptionHandler exceptionHandler,ThreadPoolExecutor executor){
//        this("default-async",exceptionHandler,executor);
//    }
//}
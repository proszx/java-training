package com.company.juc.jucdesign.future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class FutureServiceImpl<IN,OUT> implements FutureService<IN,OUT>{
    private final static String FUTURE_THREAD_PREFIX="FUTURE--";
    private final AtomicInteger nextCounter=new AtomicInteger(0);
    private String getNextName(){
        //atomic 自增
        return FUTURE_THREAD_PREFIX+nextCounter.getAndIncrement();
    }
    //将耗时操作交给一个线程去执行，从而达到异步的目的，提交线程在提交任务和获得计算结果的过程中可以进行其他的任务执行。
    @Override
    public Future<?> submit(Runnable runnable) {
        // TODO Auto-generated method stub
        //    return null;
        final FutureTask<Void> future=new FutureTask<>();
        new Thread(()->{
            runnable.run();
            future.finish(null);
        },getNextName()).start();
        return future;
    }
    @Override
    public  Future<OUT> submit(Task<IN, OUT> task, IN input, Callback<OUT> callback) {
        // TODO Auto-generated method stub
        final FutureTask<OUT> future=new FutureTask<>();
        new Thread(()->{
            OUT result=task.get(input);
            future.finish(result);
            if(null!=callback){
                callback.call(result);
            }
        },getNextName()).start();
        return future;
    }
    @Override
    public Future<OUT> submit(Task<IN, OUT> task, IN input) {
        // TODO Auto-generated method stub
        // return null;
        final FutureTask<OUT> future=new FutureTask<>();
        new Thread(()->{
            OUT result=task.get(input);
            future.finish(result);
        },getNextName()).start();
        return future;
    }
    public static void main(String[] args) throws InterruptedException {
       /** FutureService<Void,Void> service=FutureService.newService();
        Future<?> future=service.submit(()->{
            try {
                TimeUnit.SECONDS.sleep(1);
            } catch (Exception e) {
                e.printStackTrace();
                //TODO: handle exception
            }
            System.out.println("success!");
        });
        future.get();
        
        FutureService<String,Integer> service=FutureService.newService();
        String inputs="how i know the length";
        Future<Integer> future=service.submit(input->{
            try {
                TimeUnit.SECONDS.sleep(1);
            } catch (Exception e) {
                //TODO: handle exception
                e.printStackTrace();
            }
            return input.length();
        }, inputs);
        System.out.println(future.get());
         */
        FutureService<String,Integer> service=FutureService.newService();
        String inputs="how i know the length";
        Future<Integer> future=service.submit(input->{
            try {
                TimeUnit.SECONDS.sleep(1);
            } catch (Exception e) {
                //TODO: handle exception
                e.printStackTrace();
            }
            return input.length();
        }, inputs,System.out::println);
        System.out.println(future.get());
    }



}

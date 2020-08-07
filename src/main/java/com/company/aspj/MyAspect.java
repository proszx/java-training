package com.company.aspj;

import org.aspectj.lang.annotation.*;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;


@Aspect
public class MyAspect {
    Lock lock;
    Condition condition;
    MyAspect(Lock lock,Condition condition){
        this.lock=lock;
        this.condition=condition;
    }
    @Pointcut("execution(*com.company.aspj.SerivceImpl.printUser())")
    public  void pointCut(){}


    @Before("pointCut()")
    public void adviceBefore(){
        lock.lock();
        try{
            System.out.println("action before");
            condition.signalAll();
        }finally {
            lock.unlock();
        }
    }

    @After("pointCut()")
    public  void adviceAfter() throws  InterruptedException{
        lock.lock();
        try{
            condition.await();
            System.out.println("action after");
        }finally {
            lock.unlock();
        }

    }

    @AfterReturning("execution(*com.company.aspj.SerivceImpl.printUser(...))")
    public  void adviceAfterReturnning(){
        System.out.println("AfterReturning");
    }

    @AfterThrowing("execution(*com.company.aspj.SerivceImpl.printUser(...))")
    public void afterThrowing(){
        System.out.println("AfterThrowing");
    }

}

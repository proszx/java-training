package com.company.juc.jucdesign.activeobject.general;

import com.company.juc.jucdesign.activeobject.ActiveFuture;
import com.company.juc.jucdesign.future.Future;
import java.lang.reflect.Method;
public class ActiveMessage {
    private final Object[] objects;
    private final Method method;
    private final ActiveFuture<Object> future;
    private final Object service;

    static class Builder{
        private Object[] objects;
        private Method method;
        private ActiveFuture<Object> future;
        private Object service;
        public Builder useMethod(Method method){
            this.method=method;
            return this;
        }
        public Builder getFuture(ActiveFuture<Object> future){
            this.future=future;
            return this;
        }
        public Builder withObjects(Object[] objects){
            this.objects=objects;
            return this;
        }
        public Builder getService(Object service){
            this.service=service;
            return this;
        }
        public ActiveMessage build(){
            return new ActiveMessage(this);
        }
    }
    private ActiveMessage(Builder builder){
        this.objects=builder.objects;
        this.method=builder.method;
        this.future=builder.future;
        this.service=builder.service;
        
    }
    public void execute(){
        try {
            Object result=method.invoke(service, objects);
            if(future!=null){
                Future<?> real=(Future<?>) result;
                Object realresult=real.get();
                future.finish(realresult);
            }
        } catch (Exception e) {
            //TODO: handle exception
            if(future!=null){
                future.finish(null);
            }
            e.printStackTrace();
        }
    }
}
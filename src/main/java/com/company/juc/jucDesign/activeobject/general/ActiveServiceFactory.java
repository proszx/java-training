package com.company.juc.jucdesign.activeobject.general;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

import com.company.juc.jucdesign.activeobject.ActiveFuture;
import com.company.juc.jucdesign.activeobject.general.ActiveMessageQueue;
import com.company.juc.jucdesign.activeobject.OrderService;
import com.company.juc.jucdesign.activeobject.OrderServiceImpl;
import com.company.juc.jucdesign.future.Future;

public class ActiveServiceFactory {
    private final static ActiveMessageQueue queue=new ActiveMessageQueue();
    public static <T> T active(T instance){
        Object proxy=Proxy.newProxyInstance(instance.getClass().getClassLoader(), instance.getClass().getInterfaces(),new ActiveInvocationHandler<>(instance));
        return (T) proxy;
    }
    private static class ActiveInvocationHandler<T> implements InvocationHandler{
        private final T instance;

        ActiveInvocationHandler(T instance){
            this.instance=instance; 
        }
		@Override
		public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
			// TODO Auto-generated method stub
			if(method.isAnnotationPresent(ActiveMethod.class)){
                this.checkMethod(method);
                ActiveMessage.Builder builder=new ActiveMessage.Builder();
                builder.useMethod(method).withObjects(args).getService(instance);
                Object result=null;
                if(this.isReturnFutureType(method)){
                    result=new ActiveFuture<>();
                    builder.getFuture((ActiveFuture)result);
                }
                queue.offer(builder.build());
                return result;
            }else{
                return method.invoke(instance, args);
            }
        }
        
        private boolean isReturnFutureType(Method method) {
            return method.getReturnType().equals(Future.class);
        }

        private void checkMethod(Method method) throws IllegalActiveMethod {
            if(!isReturnFutureType(method)&&!isReturnVoidType(method)){
                throw new IllegalActiveMethod("the method{"+method.getName()+"return type must be void or Future");
            }
        }

        private boolean isReturnVoidType(Method method){
            return method.getReturnType().equals(Void.TYPE);
        }
        
    }
    public static void main(String[] args) throws InterruptedException {
        OrderService orderService=active(new OrderServiceImpl());
        Future<String> future=orderService.findOrferDetails(23423);
        System.out.println("finished");
        System.out.println(future.get());
        System.out.println("finished");
    }

}
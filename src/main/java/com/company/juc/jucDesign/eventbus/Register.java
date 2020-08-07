//package com.company.juc.jucdesign.eventbus;
//import java.util.concurrent.*;
//import java.lang.reflect.Method;
//import java.util.*;
//
//public class Register {
//    private final ConcurrentHashMap<String,ConcurrentLinkedQueue<Subscriber>> subsContainer=new ConcurrentHashMap<>();
//	public void bind(Object subscriber) {
//        List<Method> subscribeMethods=getSubscribeMethods(subscriber);
//        subscribeMethods.forEach(m->(tierSubscriber(subscriber,m));
//    }
//
//	private List<Method> getSubscribeMethods(Object subscriber) {
//        return null;
//    }
//
//    public void unbind(Object subscriber) {
//        subsContainer.forEach((key,queue)->queue.forEach(d->{
//            if(d.getSubscribeObject()==subscriber){
//                s.setDisable(true);
//            }
//        }));
//    }
//
//    public ConcurrentLinkedQueue<Subscriber> scanSubscriber(final String topic){
//        return subsContainer.get(topic);
//    }
//
//    private void tierSubscriber(Object subscriber,Method method){
//        final Subscribe subscribe=method.getDeclaredAnnotation(Subscribe.class);
//        String topic=subscribe.topic();
//
//
//    }
//
//}

package com.company.juc.jucdesign.activeobject;
public final class OrderServiceFactory {
    private final static ActiveMessageQueue activeQueue=new ActiveMessageQueue();
    private OrderServiceFactory(){

    }
    public static OrderService toActiveObject(OrderService orderService){
        return new OrderServiceProxy(orderService,activeQueue);
    }
    public static void main(String[] args) throws InterruptedException{
        OrderService orderService=OrderServiceFactory.toActiveObject(new OrderServiceImpl());
        orderService.order("hello",12345);
        System.out.println("successed immediately");
        Thread.currentThread().join();
    }
}
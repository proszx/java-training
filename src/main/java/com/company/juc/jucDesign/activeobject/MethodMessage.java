package com.company.juc.jucdesign.activeobject;
import java.util.Map;

public abstract class MethodMessage {
    protected final Map<String,Object> param;
    protected final OrderService orderService;
    public MethodMessage(Map<String,Object> param,OrderService orderService){
        this.param=param;
        this.orderService=orderService;
    }
    public abstract void execute();
}

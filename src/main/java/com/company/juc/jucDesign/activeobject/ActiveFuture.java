package com.company.juc.jucdesign.activeobject;

import com.company.juc.jucdesign.future.FutureTask;

public class ActiveFuture<T> extends FutureTask<T> {
    @Override
    public void finish(T result){
        super.finish(result);
    }
}

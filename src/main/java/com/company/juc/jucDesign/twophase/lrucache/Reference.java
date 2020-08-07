package com.company.juc.jucdesign.twophase.lrucache;
// weak reference
// phantom reference\
// softReference
public class Reference{
    private final byte[] data=new byte[2<<19];

    @Override
    protected void finalize() throws Throwable{
        System.out.println("GC ing");
    }
}
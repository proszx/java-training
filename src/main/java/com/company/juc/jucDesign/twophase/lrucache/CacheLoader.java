package com.company.juc.jucdesign.twophase.lrucache;

@FunctionalInterface
public interface CacheLoader<T1, T2> {
    T2 load(T1 k);
}

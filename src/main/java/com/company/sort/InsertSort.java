package com.company.sort;

public class InsertSort<T> extends Func{

    @Override
    public void sort(Comparable[] nums) {
        for(int i=1;i<nums.length;i++){
            for(int j=i;j>0&&less(nums[j],nums[j-1]);j--){
                swap(nums,j,j-1);
            }
        }
    }
}

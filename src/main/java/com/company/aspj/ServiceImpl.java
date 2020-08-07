package com.company.aspj;


@org.springframework.stereotype.Service
public class ServiceImpl implements  Service{
    public ServiceImpl(){

    }
    private volatile User user;
    @Override
    public void printUser() {
        if(user==null){
            synchronized(this){
                if(user==null) {
                    user = new User(121, "prozx");
                }
            }
        }
        System.out.print(user.getId());
        System.out.print(user.getName());

    }

    @Override
    public void printUser(User user) {
        if(this.user==null){
            synchronized (this){
                if(this.user==null){
                    this.user=user;
                }
            }
        }
        System.out.print(user.getId());
        System.out.print(user.getName());
    }
}

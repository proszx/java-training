package com.company.aspj;

public class User {

    private int id;
    private String Name;
    public User(int id,String Name){
        this.id=id;
        this.Name=Name;
    }
    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return Name;
    }

    public void setName(String name) {
        Name = name;
    }
}

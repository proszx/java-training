package stream;

import java.util.*;
public class ArrayListStream {
    public static void main(String[] args) {
        List<String> list=Arrays.asList("Java","Thread","Concurr","Scala");

        list.parallelStream().map(String::toLowerCase).forEach(System.out::println);
        list.forEach(System.out::println);
    }
}
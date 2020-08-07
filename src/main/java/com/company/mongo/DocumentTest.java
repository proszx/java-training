//package com.company.mongo;
//
//import com.mongodb.MongoClient;
//import com.mongodb.client.FindIterable;
//import com.mongodb.client.MongoCollection;
//import com.mongodb.client.MongoCursor;
//import com.mongodb.client.MongoDatabase;
//import org.bson.Document;
//import org.springframework.beans.factory.annotation.Value;
//
//import java.text.DateFormat;
//import java.util.ArrayList;
//import java.util.List;
//
//
//public class MongoTest {
//
//    @Value("47017")
//    private Integer serverPort;
//
//    @Value("202.115.16.131")
//    private String serverHost;
//
//    @Value("crawldata")
//    private String serverDatabase;
//
//
//
//    private final static Integer batches = 1000;
//
//    private MongoDatabase serverMongoDB = null;
//
//
//    private final static String localCollection = "articles";
//    private final static String remoteCollection = "articles";
//    public static boolean syncMongoData() {
//        MongoClient serverClient = new MongoClient(serverHost, serverPort);
//
//        serverMongoDB = serverClient.getDatabase(serverDatabase);
//
//        try {
//            while (serverMongoDB != null ) {
//                MongoCollection<Document> server = serverMongoDB.getCollection(remoteCollection);
//                FindIterable<Document> iter = server.find();
//                MongoCursor<Document> serverCursor = iter.iterator();
//                List<Document> documents = new ArrayList<>();
//                while (serverCursor.hasNext()) {
//                    Document document = serverCursor.next();
//                    //转换发帖日期和爬虫时间类型
//                    String createTime = document.getString("create_time");
//                    //document.remove("create_time");
//                    document.put("create_time", DateFormat.getDateTimeInstance().parse(createTime));
//                    String crawlTime = document.getString("crawl_time");
//                    //document.remove("crawl_time");
//                    document.put("crawl_time", DateFormat.getDateTimeInstance().parse(crawlTime));
//                    documents.add(document);
//
//                }
//            }
//            return true;
//        } catch (Exception e) {
//            //TODO: handle exception
//
//            return false;
//        }finally{
//
//            serverClient.close();
//        }
//    }
//
//    public static void main(String[] args) {
//        syncMongoData();
//    }
//}

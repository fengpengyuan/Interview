//package com.leetcode;
//
//import java.io.File;
//
//import android.content.Context;
//import android.database.sqlite.SQLiteDatabase;
//import android.telephony.TelephonyManager;
//import android.util.Log;
//
//public class temp {
//
//	
//	public void createDatabase(SQLiteDatabase myDB){
//    	TelephonyManager tManager = (TelephonyManager)getSystemService(Context.TELEPHONY_SERVICE);
//    	String uuid = tManager.getDeviceId();
//    	try {
//			File dbfile = new File(SQLite_NAME);
//			myDB = SQLiteDatabase.openOrCreateDatabase(dbfile, null);
//
//			/* Create a Table in the Database. */
//			myDB.execSQL("CREATE TABLE IF NOT EXISTS "
//					+ TableName
//					+ " (id INTEGER PRIMARY KEY,uuid text NULL DEFAULT "+uuid+",time datetime default current_timestamp,"
//					+ " lat real, lon real);");
//			
//			
//			/*create survey table*/
//			
//			/* Create a Table in the Database. */
//			myDB.execSQL("CREATE TABLE IF NOT EXISTS "
//					+ TableName2
//					+ " (id INTEGER PRIMARY KEY,uuid text NULL DEFAULT "+uuid+", popup datetime not null, response datetime, subtime datetime,"
//					+"lastState text, curState text,"
//					+"lat double, lon double,"
//					+ " q1 text, q2 text, q3 text, q4 text,"
//					+ " q5 text,q6 text, q7 text,q8 text);");
//
//			String CREATE_TABLE = "CREATE TABLE IF NOT EXISTS "
//					+ "Checkins"
//					+ " (checkinId TEXT PRIMARY KEY UNIQUE,uuid text NULL DEFAULT "+uuid+",createdAt INTEGER,"
//					+ " timeZoneOffset INTEGER, venueId TEXT)";
//	 
//	        // create checkin table
//	        myDB.execSQL(CREATE_TABLE);
//	        
//	        //SQL statement to create venues table
//	        CREATE_TABLE="CREATE TABLE IF NOT EXISTS "
//					+ "Venues"
//					+ " (venueId TEXT PRIMARY KEY UNIQUE,uuid text NULL DEFAULT "+uuid+",name TEXT, phone TEXT,"
//					+ " address TEXT, lat REAL, lng REAL, zipCode INTEGER, city TEXT," 
//					+ "state TEXT, country TEXT, checkinsCount INTEGER, usersCount INTEGER," 
//					+ "tipCount INTEGER, femLikes INTEGER, maleLikes INTEGER) ";
//	        // create venues table
//	        myDB.execSQL(CREATE_TABLE);
//	        
//	        //SQL statement to create categories table
//	        CREATE_TABLE="CREATE TABLE IF NOT EXISTS "
//					+ "Categories"
//					+ " (venueId TEXT PRIMARY KEY UNIQUE,uuid text NULL DEFAULT "+uuid;
//	        for(int i = 1 ; i < 6 ; i++){
//	        	CREATE_TABLE += ", cat" + i + " TEXT";
//	        }
//	        CREATE_TABLE += ")";
//	        
//	        // create venues table
//	        myDB.execSQL(CREATE_TABLE);
//	        
//	      //SQL statement to create app table
//	        CREATE_TABLE="CREATE TABLE IF NOT EXISTS "
//					+ "Applications"
//					+ " (id INTEGER PRIMARY KEY,uuid text NULL DEFAULT "+uuid+",time datetime timestamp default (datetime('now', 'localtime')), appname text, "
//					+ "type0 text, category0 text, type1 text, category1 text, type2 text, category2 text)";
//	        
//	        // create app table
//	        myDB.execSQL(CREATE_TABLE);
//	        
//	      //SQL statement to create app table
//	        CREATE_TABLE="CREATE TABLE IF NOT EXISTS "
//					+ "Emotions"
//					+ " (id INTEGER PRIMARY KEY, uuid text NULL DEFAULT "+uuid+",time datetime timestamp default (datetime('now', 'localtime')), lively text, "
//					+ "happy text, sad text, tired text, caring text, content text, gloomy text, jittery text, drowsy text, grouchy text, peppy text, nervous text, calm text, loving text, fed_up text, active text)";
//			        
//	        // create app table
//	        myDB.execSQL(CREATE_TABLE);
//	        
//		} catch (Exception e) {
//			Log.e("Error", "Error", e);
//		} finally {
//			if (myDB != null)
//				myDB.close();
//		}
//    }
//	
//	
//	
//	
//}

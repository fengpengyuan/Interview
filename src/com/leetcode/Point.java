package com.leetcode;

public class Point {
	int x;
	int y;
	Point() { x = 0; y = 0; }
	Point(int a, int b) { x = a; y = b; }
	
	public String toString(){
		return "("+x+", "+y+")";
	}
}

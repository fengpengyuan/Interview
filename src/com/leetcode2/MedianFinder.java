package com.leetcode2;

import java.util.Collections;
import java.util.PriorityQueue;

public class MedianFinder {
	PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
	PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(1, Collections.reverseOrder());

	// Adds a number into the data structure.
//	public MedianFinder() {
//		lowHalf = new PriorityQueue<Integer>(1, Collections.reverseOrder());
//		highHalf = new PriorityQueue<Integer>();
//	}
//
//	public void addNum(int num) {
//		if (lowHalf.isEmpty() || lowHalf.peek() >= num)
//			lowHalf.offer(num);
//		else
//			highHalf.offer(num);
//		if (lowHalf.size() - highHalf.size() > 1)
//			highHalf.offer(lowHalf.poll());
//		else if (highHalf.size() - lowHalf.size() > 1)
//			lowHalf.offer(highHalf.poll());
//	}
//
//	// Returns the median of current data stream
//	public double findMedian() {
//		double curMed = 0;
//		if (lowHalf.isEmpty() || highHalf.size() > lowHalf.size())
//			curMed = highHalf.peek();
//		else if (highHalf.isEmpty() || lowHalf.size() > highHalf.size())
//			curMed = lowHalf.peek();
//		else if (lowHalf.size() == highHalf.size())
//			curMed = (lowHalf.peek() + highHalf.peek()) / 2.0;
//		return curMed;
//	}
	
	// Adds a number into the data structure.
    public void addNum(int num) {
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());
        if(maxHeap.size()<minHeap.size())
        	maxHeap.offer(minHeap.poll());
    }

    // Returns the median of current data stream
    public double findMedian() {
        if(maxHeap.size()==minHeap.size()){
        	return (maxHeap.peek()+minHeap.peek())/2.0;
        }else
        	return maxHeap.peek();
    }
	
	
}

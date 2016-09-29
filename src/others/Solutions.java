package others;

import java.util.Comparator;
import java.util.PriorityQueue;

public class Solutions {
	
	public int kthSmallest(int[][] matrix, int k) {
        int m = matrix.length;
        int n = matrix[0].length;
        PriorityQueue<Pair> que=new PriorityQueue<Pair>(1, new Comparator<Pair>(){

			@Override
			public int compare(Pair o1, Pair o2) {
				// TODO Auto-generated method stub
				return o1.v-o2.v;
			}
        	
        });
        int count = 0;
        for(int i=0;i<m;i++){
        	que.add(new Pair(i, 0, matrix[i][0]));
        }
        
        while(count<k){
        	Pair p = que.poll();
        	count++;
        	if(count==k)
        		return p.v;
        	if(p.c<n-1){
        		que.offer(new Pair(p.r, p.c+1, matrix[p.r][p.c+1]));
        	}
        }
        return -1;
    }
	
	public static void main(String[] args){
		int[][] matrix={{ 1,  5,  9},
				   {10, 11, 13},
				   {12, 13, 15}};
		
		Solutions s=new Solutions();
		System.out.println(s.kthSmallest(matrix, 2));
	}
	
	

}

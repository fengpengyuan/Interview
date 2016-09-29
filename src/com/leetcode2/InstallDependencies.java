package com.leetcode2;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.PriorityQueue;

public class InstallDependencies {
	public static void main(String[] args){
        String[][] deps = {{"gcc", "gdb"},{"gcc", "visualstudio"},{"windows", "gcc"}
        , {"windows", "sublime"}, {"libc", "gcc"}, {"libc2", "gcc"}, {"unix", "cat"}
        , {"windows", "libc"}, {"windows", "libc2"}, {"linux", "cat"}, {"windows", "cat"}
        , {"solaris", "cat"}, {"macos","cat"}};
        InstallDependencies id = new InstallDependencies();
        id.install(deps, 7);
    }
    
    public void install(String[][] deps, int n){
        HashMap<String, Software> map = new HashMap<String,Software>();
        // 根据依赖关系建图
        for(String[] dep : deps){
            Software src = map.get(dep[0]);
            Software dst = map.get(dep[1]);
            if(src == null){
                src = new Software(dep[0]);
            }
            if(dst == null){
                dst = new Software(dep[1]);
            }
            src.targets.add(dst);
            dst.deps = dst.deps + 1;
            map.put(dep[0], src);
            map.put(dep[1], dst);
        }
        // 用一个优先队列来遍历我们的图
        PriorityQueue<Software> pq = new PriorityQueue<Software>(11 ,new Comparator<Software>(){
            public int compare(Software s1, Software s2){
                return s1.deps - s2.deps;
            }
        });
        for(String name : map.keySet()){
            if(map.get(name).deps == 0){
                pq.offer(map.get(name));
            }
        }
        while(!pq.isEmpty()){
            Software curr = pq.poll();
            System.out.println(curr.name);
            for(Software dst : curr.targets){
                dst.deps = dst.deps - 1;
                if(dst.deps == 0){
                    pq.offer(dst);
                }
            }
        }
    }
}

class Software{
    String name;
    int deps;
    ArrayList<Software> targets;
    public Software(String name){
        this.deps = 0;
        this.name = name;
        this.targets = new ArrayList<Software>();
    }
}

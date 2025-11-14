#!/usr/sbin/dtrace -s
/*
 * DTrace script to trace Accelerate framework calls
 * Usage: sudo dtrace -s trace_accelerate_calls.d -p <python_pid>
 */

pid$target:libAccelerate::entry
{
    @funcs[probefunc] = count();
}

pid$target:libAccelerate::return
{
    self->start = timestamp;
}

pid$target:libAccelerate::return
/self->start/
{
    @times[probefunc] = sum(timestamp - self->start);
    @count[probefunc] = count();
    self->start = 0;
}

END
{
    printf("\n=== Accelerate Function Call Summary ===\n\n");
    printf("%-40s %10s %20s\n", "Function", "Count", "Total Time (ns)");
    printf("%-40s %10s %20s\n", "----------------------------------------", "----------", "--------------------");
    printa("%-40s %10@d %20@d\n", @funcs);
    printf("\n");
    printf("%-40s %10s %20s %20s\n", "Function", "Count", "Total Time (ns)", "Avg Time (ns)");
    printf("%-40s %10s %20s %20s\n", "----------------------------------------", "----------", "--------------------", "--------------------");
    printa("%-40s %10@d %20@d %20@d\n", @count, @times, @times/@count);
}


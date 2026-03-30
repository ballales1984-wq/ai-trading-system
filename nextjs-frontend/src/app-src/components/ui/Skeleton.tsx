interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = '' }: SkeletonProps) {
  return <div className={`skeleton ${className}`} />;
}

export function CardSkeleton() {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-5 w-5 rounded-full" />
      </div>
      <Skeleton className="h-7 w-32 mb-1" />
      <Skeleton className="h-4 w-16" />
    </div>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <th key={i} className="py-3 px-4">
                <Skeleton className="h-4 w-16" />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, i) => (
            <tr key={i} className="border-b border-border/50">
              {[1, 2, 3, 4, 5, 6].map((j) => (
                <td key={j} className="py-3 px-4">
                  <Skeleton className="h-4 w-20" />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <Skeleton className="h-6 w-32 mb-4" />
      <div className="h-64 flex items-end gap-1">
        {Array.from({ length: 30 }).map((_, i) => (
          <div
            key={i}
            className="flex-1 bg-primary/20 rounded-t"
            style={{ height: `${Math.random() * 80 + 20}%` }}
          />
        ))}
      </div>
    </div>
  );
}

export function DashboardSkeleton() {
  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <Skeleton className="h-8 w-48 mb-2" />
        <Skeleton className="h-4 w-64" />
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[1, 2, 3, 4].map((i) => (
          <CardSkeleton key={i} />
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ChartSkeleton />
        <div className="bg-surface border border-border rounded-lg p-4">
          <Skeleton className="h-6 w-40 mb-4" />
          <div className="grid grid-cols-2 gap-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="p-3 bg-background rounded-lg">
                <Skeleton className="h-4 w-24 mb-2" />
                <Skeleton className="h-6 w-16" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-surface border border-border rounded-lg p-4">
        <Skeleton className="h-6 w-40 mb-4" />
        <TableSkeleton />
      </div>
    </div>
  );
}

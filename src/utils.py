from rich.progress import track, Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    MofNCompleteColumn(),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
)

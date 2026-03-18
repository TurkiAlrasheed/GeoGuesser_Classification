document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("guess-form");
    if (!form) return;

    let submitted = false;
    const buttons = form.querySelectorAll(".city-btn");

    buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
            if (submitted) return;
            submitted = true;

            buttons.forEach((b) => {
                if (b !== btn) {
                    b.style.opacity = "0.4";
                    b.style.pointerEvents = "none";
                }
            });
            btn.style.borderColor = "var(--accent)";
            btn.style.background = "var(--bg-card-hover)";
        });
    });
});

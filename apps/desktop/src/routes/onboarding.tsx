import { createFileRoute } from "@tanstack/react-router";

import { OnboardingView } from "@/features/onboarding/onboarding-view";

export const Route = createFileRoute("/onboarding")({
  component: OnboardingView,
});
